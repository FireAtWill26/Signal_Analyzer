"""analyzer.py — factor signal analysis: returns, holds, statistics.

Provides:
  - compute_portfolio_returns: unified dispatcher for portfolio return methods
  - compute_signal_weighted_returns: demean + scale to target_gross (default)
  - compute_power_returns: rank → demean → scale to target_gross
  - compute_zz3000_benchmark_returns: ZZ3000 index-weighted benchmark returns
  - compute_monthly_returns: aggregate daily returns to monthly returns
  - compute_signal_statistics: mean / std / skew / kurtosis per month
  - compute_cumulative_returns: cumulative return curve
  - compute_daily_ic: daily rank IC between signal and next-day return

隔日收益统一使用 compute_return.py 中的公式：
    basePrice = close[didx-1, 0:nStock, tidx] * adjFactor[didx, 0:nStock]
    returns   = close[didx,   0:nStock, tidx] / basePrice - 1
其中 close 来自 IntervalFull.close (n_days, n_stocks, n_intv)，adjFactor 来自
CAX/adjfactor.N。tidx 为 intv 索引，intv 0-48 直接对应 tidx。

intv 回退逻辑：
    用户选择 intv=X，如果信号文件中存在 intv=X，则使用该信号；
    如果不存在，则使用小于 X 的最大的可用 intv 的信号。
    收益计算始终使用 tidx=X。
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats as sp_stats
from concurrent.futures import ThreadPoolExecutor

from data_loader import (
    load_uid, load_dates, load_universe_mask, load_close_prices,
    load_interval_close, load_adj_factor, load_zz3000_weights,
    load_dsrt_returns, load_signal_npz, load_limit_prices,
    date_to_index, month_key, month_label
)


# ── constants ─────────────────────────────────────────────────────────────────

N_STOCKS = 5860
N_INTERVALS = 49  # IntervalFull has 49 intervals (0-48)


# ── helpers ───────────────────────────────────────────────────────────────────

def _filter_valid(pred, stocks, valid_tag):
    """Return (stocks, pred) with only valid entries (vValidTag + finite)."""
    if valid_tag is not None:
        mask = valid_tag & np.isfinite(pred)
    else:
        mask = np.isfinite(pred)
    return stocks[mask], pred[mask]


def _build_stock_to_index(uid_array):
    """Build stock code → column index mapping.

    uid_array is the whole-universe stock codes (len 5459). Stock at position i
    in uid corresponds to column i in the (n_days, 5860) price/universe files.

    Returns a dict for backward compatibility. For vectorized lookup, use
    _build_stock_lookup instead.
    """
    return {s: i for i, s in enumerate(uid_array)}


def _build_stock_lookup(uid_array):
    """Build sorted uid lookup for vectorized searchsorted mapping.

    Returns:
        (sorted_uids, sort_idx) where sorted_uids is uid_array sorted
        lexicographically, and sort_idx[i] is the original column index
        of the i-th sorted uid.
    """
    sort_idx = np.argsort(uid_array, kind='stable')
    sorted_uids = uid_array[sort_idx]
    return sorted_uids, sort_idx


def _map_signal_to_universe(stocks, pred, stock_to_idx):
    """Map signal (stocks, pred) onto whole-universe column positions.

    Returns a (N_STOCKS,) array with pred values at matched positions,
    NaN elsewhere.
    """
    aligned = np.full(N_STOCKS, np.nan, dtype=np.float32)
    # Vectorized lookup: build index array, then fancy-index
    indices = np.array(
        [stock_to_idx.get(s, -1) for s in stocks], dtype=np.int64
    )
    valid = indices >= 0
    aligned[indices[valid]] = pred[valid]
    return aligned


def _map_signal_to_universe_vec(stocks, pred, sorted_uids, sort_idx):
    """Vectorized version of _map_signal_to_universe using searchsorted.

    Args:
        stocks: array of stock codes from the signal file.
        pred: array of signal values, aligned with stocks.
        sorted_uids: sorted uid array from _build_stock_lookup.
        sort_idx: original column indices from _build_stock_lookup.

    Returns:
        (N_STOCKS,) array with pred values at matched positions, NaN elsewhere.
    """
    aligned = np.full(N_STOCKS, np.nan, dtype=np.float32)

    # searchsorted gives the insertion position; check for exact match
    pos = np.searchsorted(sorted_uids, stocks)
    valid = (pos < len(sorted_uids)) & (sorted_uids[pos] == stocks)

    if valid.any():
        orig_indices = sort_idx[pos[valid]]
        aligned[orig_indices] = pred[valid]

    return aligned


def _intv_value_from_path(filepath):
    """Extract intv value (integer 0-48) from a signal npz filepath.

    File naming: alpha.YYYYMMDD.intv49i<XX>.npz
    The intv value is <XX> (e.g. i00 → 0, i36 → 36).
    Returns None if the file is not a valid intv signal file.
    """
    bn = filepath.split('/')[-1]
    parts = bn.split('.')
    if len(parts) < 4 or parts[0] != 'alpha':
        return None
    intv_token = parts[2]  # intv49i00
    if not intv_token.startswith('intv'):
        return None
    rest = intv_token[4:]  # '49i00'
    if 'i' not in rest:
        return None
    idx = rest.index('i')
    intv_str = rest[idx + 1:]  # '00'
    if not intv_str.isdigit():
        return None
    return int(intv_str)


def _find_best_intv(available_intvs, target_intv):
    """Find the best available intv for a given target intv.

    If the target intv exists in available_intvs, return it.
    Otherwise, return the largest available intv that is smaller than target.
    If no such intv exists, return None.

    Args:
        available_intvs: list of available intv values.
        target_intv: the desired intv value.

    Returns:
        The best matching intv value, or None.
    """
    if target_intv in available_intvs:
        return target_intv
    # Find the largest intv < target
    candidates = [v for v in available_intvs if v < target_intv]
    if candidates:
        return max(candidates)
    return None


def _load_signal_aligned(fp, stock_to_idx):
    """Load a signal npz file and align it to whole-universe columns.

    Returns a (N_STOCKS,) array with signal values at matched positions,
    NaN elsewhere.
    """
    sig = load_signal_npz(fp)
    stocks, pred = _filter_valid(sig['alphaV1'], sig['vStockCode'],
                                 sig['vValidTag'])
    return _map_signal_to_universe(stocks, pred, stock_to_idx)


_SIGNAL_MATRIX_CACHE = {}


def _load_signal_matrix(signal_files, sorted_uids, sort_idx, dates_arr):
    """Load all signals for an intv into a (n_days, N_STOCKS) matrix.

    This is the vectorized batch loader: it iterates over all signal files
    once, aligns each to whole-universe columns, and places it at the
    corresponding day index.

    Results are cached at module level keyed by the signal files, so repeated
    calls across different universes (which share the same signal data) don't
    re-load the npz files.

    Args:
        signal_files: list of (date_int, filepath) tuples.
        sorted_uids: sorted uid array from _build_stock_lookup.
        sort_idx: original column indices from _build_stock_lookup.
        dates_arr: trading dates array.

    Returns:
        np.ndarray (n_days, N_STOCKS) of float32, NaN where no signal.
    """
    ck = (tuple(signal_files),)
    if ck in _SIGNAL_MATRIX_CACHE:
        return _SIGNAL_MATRIX_CACHE[ck]

    n_days = len(dates_arr)
    signal_matrix = np.full((n_days, N_STOCKS), np.nan, dtype=np.float32)

    by_date = _group_files_by_date(signal_files)

    # Load files in parallel across dates
    def _load_one(date_int_fps):
        date_int, fps = date_int_fps
        day_idx = date_to_index(date_int, dates_arr)
        if day_idx < 0:
            return None
        aligned_list = [
            _load_signal_aligned_vec(fp, sorted_uids, sort_idx)
            for fp in fps
        ]
        if not aligned_list:
            return None
        stacked = np.stack(aligned_list, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return day_idx, np.nanmean(stacked, axis=0)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_load_one, by_date.items()))

    for r in results:
        if r is not None:
            day_idx, avg = r
            signal_matrix[day_idx] = avg

    _SIGNAL_MATRIX_CACHE[ck] = signal_matrix
    return signal_matrix


def _load_signal_aligned_vec(fp, sorted_uids, sort_idx):
    """Vectorized signal loader using searchsorted on sorted uid strings.

    Args:
        fp: signal npz file path.
        sorted_uids: sorted uid array from _build_stock_lookup.
        sort_idx: original column indices from _build_stock_lookup.

    Returns:
        (N_STOCKS,) float32 array with signal at matched columns, NaN elsewhere.
    """
    d = np.load(fp, allow_pickle=False)
    alpha = d['alphaV1']
    stocks = d['vStockCode']
    valid_tag = d['vValidTag'] if 'vValidTag' in d.files else None

    # Filter valid: vValidTag (if present) + finite alpha
    if valid_tag is not None:
        mask = valid_tag & np.isfinite(alpha)
    else:
        mask = np.isfinite(alpha)
    stocks = stocks[mask]
    pred = alpha[mask]

    # Vectorized mapping via string searchsorted
    aligned = np.full(N_STOCKS, np.nan, dtype=np.float32)
    pos = np.searchsorted(sorted_uids, stocks)
    valid = (pos < len(sorted_uids)) & (sorted_uids[pos] == stocks)
    if valid.any():
        orig_indices = sort_idx[pos[valid]]
        aligned[orig_indices] = pred[valid]

    return aligned


def compute_next_day_returns(
    close_intervals, adj_factor, day_idx, tidx, n_stocks=N_STOCKS
):
    """Compute next-day return ratios per the formula in compute_return.py.

        basePrice   = close[didx-1, 0:nStock, tidx] * adjFactor[didx, 0:nStock]
        returnRatio = close[didx,   0:nStock, tidx] / basePrice - 1

    The adjFactor here is the forward-adjustment factor from CAX/adjfactor.N:
    values are 1.0 on normal days and differ from 1.0 only on ex-dividend /
    ex-split dates, where it adjusts the previous day's close to be on the
    same scale as today's close.

    Args:
        close_intervals: (n_days, n_stocks, n_intv) float32, from IntervalFull.close.
        adj_factor: (n_days, n_stocks) float32, from CAX/adjfactor.N.
        day_idx: date index (didx).
        tidx: interval index.
        n_stocks: number of stocks to read (first n_stocks columns).

    Returns:
        np.ndarray (n_stocks,) of float32 return ratios.
    """
    base_price = close_intervals[day_idx - 1, :n_stocks, tidx].copy()
    base_price *= adj_factor[day_idx, :n_stocks]
    return_ratio = close_intervals[day_idx, :n_stocks, tidx] / base_price - 1
    return return_ratio


def _load_returns(data_root, return_source='raw'):
    """Load return data based on return_source.

    Args:
        data_root: root of data directory.
        return_source: 'raw' (close-based, requires adj_factor) or 'dsrt'
            (pre-computed DSRT returns from CNE5Ret).

    Returns:
        For 'raw': (close_intervals, adj_factor) tuple.
        For 'dsrt': (dsrt_returns, None) tuple.
    """
    if return_source == 'dsrt':
        return load_dsrt_returns(data_root), None
    # default: raw
    return load_interval_close(data_root), load_adj_factor(data_root)


def _get_next_day_returns(
    returns_data, adj_factor, day_idx, tidx, return_source='raw',
    n_stocks=N_STOCKS,
):
    """Get next-day returns for the given day and interval index.

    Args:
        returns_data: either close_intervals (for 'raw') or dsrt_returns
            (for 'dsrt').
        adj_factor: adj_factor array (only used for 'raw').
        day_idx: date index (didx).
        tidx: interval index.
        return_source: 'raw' or 'dsrt'.
        n_stocks: number of stocks to read.

    Returns:
        np.ndarray (n_stocks,) of return ratios.
    """
    if return_source == 'dsrt':
        # DSRT returns are pre-computed; just slice
        return returns_data[day_idx, :n_stocks, tidx].copy()
    # default: raw (close-based)
    return compute_next_day_returns(
        returns_data, adj_factor, day_idx, tidx, n_stocks=n_stocks
    )


_RETURNS_ALL_CACHE = {}


def _rank_ic_fast(signal, returns):
    """Compute Spearman rank IC using np.argsort (fast, no scipy overhead).

    Args:
        signal: 1D array of signal values.
        returns: 1D array of return values, same length as signal.

    Returns:
        float IC value (NaN if undefined).
    """
    # argsort twice gives ordinal ranks (0..n-1)
    rs = np.argsort(np.argsort(signal)).astype(np.float64)
    rr = np.argsort(np.argsort(returns)).astype(np.float64)
    rs -= rs.mean()
    rr -= rr.mean()
    denom = np.sqrt((rs * rs).sum() * (rr * rr).sum())
    if denom > 0:
        return float((rs * rr).sum() / denom)
    return np.nan


def _batch_rank_ic(signal_matrix, returns_matrix, valid_mask):
    """Compute daily rank IC (Spearman) for all days, vectorized.

    Avoids the per-day overhead of ``sp_stats.spearmanr`` by computing
    ranks inline with ``argsort`` and then computing the Pearson
    correlation of the ranks.

    Args:
        signal_matrix: (n_days, N_STOCKS) float32.
        returns_matrix: (n_days, N_STOCKS) float32.
        valid_mask: (n_days, N_STOCKS) bool.

    Returns:
        np.ndarray (n_days,) of float64 IC values (NaN where < 10 valid).
    """
    n_days = signal_matrix.shape[0]
    ics = np.full(n_days, np.nan, dtype=np.float64)

    for d in range(n_days):
        m = valid_mask[d]
        n = int(m.sum())
        if n < 10:
            continue
        s = signal_matrix[d, m]
        r = returns_matrix[d, m]
        rs = sp_stats.rankdata(s)
        rr = sp_stats.rankdata(r)
        rs -= rs.mean()
        rr -= rr.mean()
        denom = np.sqrt((rs * rs).sum() * (rr * rr).sum())
        if denom > 0:
            ics[d] = float((rs * rr).sum() / denom)

    return ics


def _precompute_all_returns(data_root, return_source='raw'):
    """Precompute next-day returns for ALL tidx values at once (vectorized).

    Returns:
        (n_days, n_stocks, n_intv) float32 array. returns[d, s, t] is the
        return for day d, stock s, interval t. returns[0] is NaN.
    """
    ck = ('all_returns', data_root, return_source)
    if ck in _RETURNS_ALL_CACHE:
        return _RETURNS_ALL_CACHE[ck]

    if return_source == 'dsrt':
        returns = load_dsrt_returns(data_root)
        _RETURNS_ALL_CACHE[ck] = returns
        return returns

    # raw: close-based, fully vectorized over (days, stocks, intervals)
    close = load_interval_close(data_root)  # (n_days, n_stocks, n_intv)
    adj = load_adj_factor(data_root)         # (n_days, n_stocks)

    n_days, n_stocks, n_intv = close.shape
    returns = np.full_like(close, np.nan, dtype=np.float32)

    if n_days > 1:
        # prev[d, s, t] = close[d, s, t]; we need close[d-1] * adj[d]
        # Vectorized: base[d-1, s, t] = close[d-1, s, t] * adj[d, s]
        prev = close[:-1]                    # (n_days-1, n_stocks, n_intv)
        curr = close[1:]                     # (n_days-1, n_stocks, n_intv)
        adj_b = adj[1:, :, None]             # (n_days-1, n_stocks, 1)
        base = prev * adj_b                  # broadcast over intervals
        returns[1:] = curr / base - 1

    _RETURNS_ALL_CACHE[ck] = returns
    return returns


def _precompute_returns_matrix(
    returns_data, adj_factor, tidx, return_source='raw', n_stocks=N_STOCKS
):
    """Precompute next-day returns for all days at a given tidx.

    Returns a (n_days, n_stocks) float32 array where entry [d, s] is the
    return for day d. returns_matrix[0] is NaN (no previous day).

    For 'raw':
        base_price = close[d-1, :, tidx] * adj_factor[d, :]
        return = close[d, :, tidx] / base_price - 1
    For 'dsrt':
        pre-computed returns, just slice for tidx.
    """
    if return_source == 'dsrt':
        return returns_data[:, :n_stocks, tidx].copy()

    close_tidx = returns_data[:, :n_stocks, tidx]  # (n_days, n_stocks)
    n_days = close_tidx.shape[0]

    returns_matrix = np.full((n_days, n_stocks), np.nan, dtype=np.float32)

    if n_days <= 1:
        return returns_matrix

    prev_close = close_tidx[:-1]   # (n_days-1, n_stocks), days 0..n_days-2
    curr_close = close_tidx[1:]    # (n_days-1, n_stocks), days 1..n_days-1
    curr_adj = adj_factor[1:, :n_stocks]  # (n_days-1, n_stocks)

    base_prices = prev_close * curr_adj
    returns_matrix[1:] = curr_close / base_prices - 1

    return returns_matrix


def _limit_hit_mask(
    close_intervals, up_lim, dn_lim, day_idx, tidx, n_stocks=N_STOCKS,
):
    """Boolean mask of stocks at limit-up or limit-down at day_idx, tidx.

    A stock is considered to have hit the limit if its interval close price
    at the given tidx is at (or beyond) the daily limit-up or limit-down
    price. Such stocks are illiquid (locked at the limit) and should be
    excluded from the tradeable universe when ``exclude_limit=True``.

    Args:
        close_intervals: (n_days, n_stocks, n_intv) close prices.
        up_lim: (n_days, n_stocks) daily limit-up prices.
        dn_lim: (n_days, n_stocks) daily limit-down prices.
        day_idx: trading day index (the day the position is held).
        tidx: interval index at which to check the limit.
        n_stocks: number of stocks to read.

    Returns:
        np.ndarray (n_stocks,) of bool. True = stock is at limit (exclude).
    """
    if close_intervals is None or up_lim is None or dn_lim is None:
        return np.zeros(n_stocks, dtype=bool)

    price = close_intervals[day_idx, :n_stocks, tidx]
    up = up_lim[day_idx, :n_stocks]
    dn = dn_lim[day_idx, :n_stocks]

    tol = 1e-4
    hit_up = (
        np.isfinite(price) & np.isfinite(up) & (up > 0)
        & (price >= up - tol)
    )
    hit_dn = (
        np.isfinite(price) & np.isfinite(dn) & (dn > 0)
        & (price <= dn + tol)
    )
    return hit_up | hit_dn


def _load_limit_filter(data_root, return_source, returns_data=None):
    """Load limit prices and close intervals for limit filtering.

    Limit checking always uses close prices, so we always load
    ``load_interval_close``. The ``returns_data`` argument is ignored
    (kept for backwards compatibility with existing callers).

    Returns:
        (close_for_limit, up_lim, dn_lim) tuple.
    """
    up_lim, dn_lim = load_limit_prices(data_root)
    close_for_limit = load_interval_close(data_root)
    return close_for_limit, up_lim, dn_lim



def _select_signal_files_by_intv(by_intv, target_intv):
    """Select signal files for the best matching intv.

    Uses the fallback logic: if target_intv exists, use it; otherwise use
    the largest available intv smaller than target.

    Args:
        by_intv: dict {intv_value: [(date, path), ...]} from scan_signal_files.
        target_intv: the user-selected intv value.

    Returns:
        Tuple of (best_intv, signal_files) where signal_files is a list of
        (date, path) tuples. Returns (None, []) if no suitable intv found.
    """
    available = sorted(by_intv.keys())
    best_intv = _find_best_intv(available, target_intv)
    if best_intv is None:
        return None, []
    return best_intv, by_intv[best_intv]


def _group_files_by_date(signal_files):
    """Group signal files by date.

    Args:
        signal_files: list of (date_int, filepath) tuples.

    Returns:
        dict: {date_int: [filepath, ...]} sorted by date.
    """
    by_date = {}
    for date_int, fp in signal_files:
        by_date.setdefault(date_int, []).append(fp)
    return by_date


# ── signal-weighted long-short portfolio returns ──────────────────────────────

def compute_signal_weighted_returns(
    signal_files_by_intv,
    universe_key,
    data_root,
    target_gross=20e6,
    intv=0,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily returns of signal-weighted long-short portfolio.

    Position sizing:
      1. Demean the signal (subtract mean) so positions sum to 0.
      2. Scale absolute values so sum of |positions| = target_gross.
      3. Position_i = adjusted signal value (in currency).

    Daily PnL = sum(position_i * return_i)
    Daily return = PnL / target_gross

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        target_gross: target gross exposure (sum of |positions|). Default 20e6.
        intv: the interval index to use for return computation. Default 0.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['date', 'portfolio_return', 'n_stocks', 'gross'].
    """
    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)

    uni_mask = load_universe_mask(universe_key, data_root)
    returns_data, adj_factor = _load_returns(data_root, return_source)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, returns_data
    ) if exclude_limit else (None, None, None)

    # Select signal files using fallback logic
    best_intv, signal_files = _select_signal_files_by_intv(
        signal_files_by_intv, intv
    )
    if best_intv is None:
        return pd.DataFrame()

    by_date = _group_files_by_date(signal_files)

    # Precompute returns matrix for tidx = intv + delay
    tidx = intv + delay
    returns_matrix = _precompute_returns_matrix(
        returns_data, adj_factor, tidx,
        return_source=return_source,
    )

    daily_returns = []
    for date_int in sorted(by_date.keys()):
        day_idx = date_to_index(date_int, dates_arr)
        if day_idx < 0 or day_idx + 1 >= len(dates_arr):
            continue

        aligned_signals = []
        for fp in by_date[date_int]:
            aligned = _load_signal_aligned_vec(fp, sorted_uids, sort_idx)
            aligned_signals.append(aligned)

        if not aligned_signals:
            continue

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', category=RuntimeWarning)
            stacked = np.stack(aligned_signals, axis=0)
            if stacked.size == 0:
                continue
            avg_signal = np.nanmean(stacked, axis=0)

        # Get precomputed returns for day_idx + 1
        returns = returns_matrix[day_idx + 1]

        # Exclude limit-hit stocks at trading tidx on day_idx+1
        if exclude_limit:
            limit_mask = _limit_hit_mask(
                close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
            )
            returns = np.where(limit_mask, np.nan, returns)

        # Apply universe mask
        membership = uni_mask[day_idx]
        valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
        if valid_mask.sum() < 10:
            continue

        signal_valid = avg_signal[valid_mask]
        returns_valid = returns[valid_mask]

        # Step 1: Demean so positions sum to 0
        signal_demeaned = signal_valid - np.mean(signal_valid)

        # Step 2: Scale so sum of |positions| = target_gross
        abs_sum = float(np.sum(np.abs(signal_demeaned)))
        if abs_sum < 1e-8:
            continue
        positions = signal_demeaned * (target_gross / abs_sum)

        # Daily PnL = sum(position * return); return = PnL / target_gross
        pnl = float(np.sum(positions * returns_valid))
        portfolio_return = pnl / target_gross

        daily_returns.append({
            'date': date_int,
            'portfolio_return': portfolio_return,
            'n_stocks': int(valid_mask.sum()),
            'gross': target_gross,
        })

    return pd.DataFrame(daily_returns)


# ── power (rank-based) portfolio returns ──────────────────────────────────────

def compute_power_returns(
    signal_files_by_intv,
    universe_key,
    data_root,
    target_gross=20e6,
    intv=0,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily returns using rank-based position sizing.

    Position sizing:
      1. Rank the signal values (1 = smallest, N = largest).
      2. Demean the ranks so positions sum to 0.
      3. Scale absolute values so sum of |positions| = target_gross.
      4. Position_i = adjusted rank value (in currency).

    This "power" method reduces the influence of signal outliers by using
    ranks instead of raw values.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        target_gross: target gross exposure. Default 20e6.
        intv: the interval index to use for return computation. Default 0.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['date', 'portfolio_return', 'n_stocks', 'gross'].
    """
    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)

    uni_mask = load_universe_mask(universe_key, data_root)
    returns_data, adj_factor = _load_returns(data_root, return_source)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, returns_data
    ) if exclude_limit else (None, None, None)

    # Select signal files using fallback logic
    best_intv, signal_files = _select_signal_files_by_intv(
        signal_files_by_intv, intv
    )
    if best_intv is None:
        return pd.DataFrame()

    by_date = _group_files_by_date(signal_files)

    # Precompute returns matrix for tidx = intv + delay
    tidx = intv + delay
    returns_matrix = _precompute_returns_matrix(
        returns_data, adj_factor, tidx,
        return_source=return_source,
    )

    daily_returns = []
    for date_int in sorted(by_date.keys()):
        day_idx = date_to_index(date_int, dates_arr)
        if day_idx < 0 or day_idx + 1 >= len(dates_arr):
            continue

        aligned_signals = []
        for fp in by_date[date_int]:
            aligned = _load_signal_aligned_vec(fp, sorted_uids, sort_idx)
            aligned_signals.append(aligned)

        if not aligned_signals:
            continue

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', category=RuntimeWarning)
            stacked = np.stack(aligned_signals, axis=0)
            if stacked.size == 0:
                continue
            avg_signal = np.nanmean(stacked, axis=0)

        # Get precomputed returns for day_idx + 1
        returns = returns_matrix[day_idx + 1]

        # Exclude limit-hit stocks at trading tidx on day_idx+1
        if exclude_limit:
            limit_mask = _limit_hit_mask(
                close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
            )
            returns = np.where(limit_mask, np.nan, returns)

        membership = uni_mask[day_idx]
        valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
        if valid_mask.sum() < 10:
            continue

        signal_valid = avg_signal[valid_mask]
        returns_valid = returns[valid_mask]

        # Step 1: Rank signal (1=smallest, N=largest)
        ranks = sp_stats.rankdata(signal_valid)  # float64

        # Step 2: Demean ranks so positions sum to 0
        ranks_demeaned = ranks - np.mean(ranks)

        # Step 3: Scale so sum of |positions| = target_gross
        abs_sum = float(np.sum(np.abs(ranks_demeaned)))
        if abs_sum < 1e-8:
            continue
        positions = ranks_demeaned * (target_gross / abs_sum)

        # Daily PnL = sum(position * return); return = PnL / target_gross
        pnl = float(np.sum(positions * returns_valid))
        portfolio_return = pnl / target_gross

        daily_returns.append({
            'date': date_int,
            'portfolio_return': portfolio_return,
            'n_stocks': int(valid_mask.sum()),
            'gross': target_gross,
        })

    return pd.DataFrame(daily_returns)


# ── ZZ3000 benchmark returns ──────────────────────────────────────────────────

def compute_zz3000_benchmark_returns(
    data_root,
    intv=0,
    delay=1,
    target_gross=20e6,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily returns of the ZZ3000 benchmark.

    ZZ3000 index weights are all positive (they are index component weights).
    The benchmark return is the long-only weighted average of next-day returns:

        benchmark_return = sum(w_i * r_i) / sum(w_i)

    This gives the true ZZ3000 index return, not a long-short portfolio.

    The benchmark always uses tidx=48 (the last 5-min interval, representing
    the close price) regardless of the factor's intv/delay. This ensures the
    benchmark is consistent across different factors.

    Args:
        data_root: root of data directory.
        intv: ignored (kept for API compatibility). The benchmark uses tidx=48.
        delay: ignored (kept for API compatibility). The benchmark uses tidx=48.
        target_gross: target gross exposure. Default 20e6.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            close (tidx=48) on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['date', 'portfolio_return', 'n_stocks', 'gross'].
    """
    dates_arr = np.array(load_dates(data_root))
    returns_data, adj_factor = _load_returns(data_root, return_source)
    zz3000_weights = load_zz3000_weights(data_root)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, returns_data
    ) if exclude_limit else (None, None, None)

    # Always use tidx=48 (close price) for the benchmark, regardless of intv/delay
    tidx = 48

    daily_returns = []
    for day_idx in range(0, len(dates_arr) - 1):
        weights = zz3000_weights[day_idx]
        ret = _get_next_day_returns(
            returns_data, adj_factor, day_idx + 1, tidx,
            return_source=return_source,
        )

        # Exclude limit-hit stocks at close (tidx=48) on day_idx+1
        # if exclude_limit:
        #     limit_mask = _limit_hit_mask(
        #         close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
        #     )
        #     ret = np.where(limit_mask, np.nan, ret)

        valid_mask = np.isfinite(weights) & (weights > 0) & np.isfinite(ret)
        if valid_mask.sum() < 10:
            continue

        w_valid = weights[valid_mask]
        r_valid = ret[valid_mask]

        # Long-only weighted average return
        portfolio_return = float(np.sum(w_valid * r_valid) / np.sum(w_valid))

        daily_returns.append({
            'date': int(dates_arr[day_idx]),
            'portfolio_return': portfolio_return,
            'n_stocks': int(valid_mask.sum()),
            'gross': target_gross,
        })

    return pd.DataFrame(daily_returns)


# ── unified portfolio returns dispatcher ──────────────────────────────────────

def compute_portfolio_returns(
    signal_files_by_intv,
    universe_key,
    data_root,
    position_method='signal_weighted',
    target_gross=20e6,
    intv=0,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Dispatch to the appropriate portfolio return computation.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        position_method: 'signal_weighted' (default) or 'power'.
            - 'signal_weighted': demean signal, scale |positions| to target_gross.
            - 'power': rank signal, then demean + scale.
        target_gross: target gross exposure.
        intv: the interval index to use for return computation. Default 0.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with daily portfolio returns.
    """
    if position_method == 'signal_weighted':
        return compute_signal_weighted_returns(
            signal_files_by_intv, universe_key, data_root,
            target_gross=target_gross, intv=intv, delay=delay,
            return_source=return_source, exclude_limit=exclude_limit,
        )
    elif position_method == 'power':
        return compute_power_returns(
            signal_files_by_intv, universe_key, data_root,
            target_gross=target_gross, intv=intv, delay=delay,
            return_source=return_source, exclude_limit=exclude_limit,
        )
    else:
        raise ValueError(f"Unknown position_method: {position_method}")


# ── all-intv averaged returns ────────────────────────────────────────────────

def compute_portfolio_returns_all_intv(
    signal_files_by_intv,
    universe_key,
    data_root,
    position_method='signal_weighted',
    target_gross=20e6,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily returns averaged across all available intv values.

    For each intv value present in signal_files_by_intv, compute the daily
    portfolio return using that intv's signal and that intv's return. Then
    align by date and average the daily returns across intv values.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        position_method: 'signal_weighted' (default) or 'power'.
        target_gross: target gross exposure.
        delay: number of intervals after signal to trade. Default 1.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['date', 'portfolio_return', 'n_stocks', 'gross'].
    """
    available_intvs = sorted(signal_files_by_intv.keys())
    if not available_intvs:
        return pd.DataFrame()

    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)
    uni_mask = load_universe_mask(universe_key, data_root)

    # Precompute ALL returns at once (vectorized over tidx). Cached at module
    # level so subsequent calls / universes reuse the same loaded data.
    all_returns = _precompute_all_returns(data_root, return_source)
    n_days = len(dates_arr)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, all_returns
    ) if exclude_limit else (None, None, None)

    # Batch-load signal matrices for all intvs in parallel (cached at module
    # level, so subsequent universes reuse the same loaded data).
    def _load_one_intv(intv):
        best_intv, signal_files = _select_signal_files_by_intv(
            signal_files_by_intv, intv
        )
        if best_intv is None:
            return None
        return intv, _load_signal_matrix(
            signal_files, sorted_uids, sort_idx, dates_arr
        )

    n_workers_load = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers_load) as ex:
        load_results = list(ex.map(_load_one_intv, available_intvs))

    signal_matrices = {}
    for r in load_results:
        if r is not None:
            signal_matrices[r[0]] = r[1]

    # Process each intv in parallel — portfolio return computation is CPU-bound
    # numpy work that releases the GIL, so a thread pool gives near-linear
    # speedup across the 47 intv variants of factor B.
    def _compute_one(intv):
        if intv not in signal_matrices:
            return None
        signal_matrix = signal_matrices[intv]
        tidx = intv + delay
        if tidx >= all_returns.shape[2]:
            return None

        daily_returns = []
        for day_idx in range(n_days - 1):
            if not np.isfinite(signal_matrix[day_idx]).any():
                continue
            date_int = int(dates_arr[day_idx])

            avg_signal = signal_matrix[day_idx]
            returns = all_returns[day_idx + 1, :, tidx]

            if exclude_limit:
                limit_mask = _limit_hit_mask(
                    close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
                )
                returns = np.where(limit_mask, np.nan, returns)

            membership = uni_mask[day_idx]
            valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
            n_valid = int(valid_mask.sum())
            if n_valid < 10:
                continue

            signal_valid = avg_signal[valid_mask]
            returns_valid = returns[valid_mask]

            if position_method == 'signal_weighted':
                signal_demeaned = signal_valid - np.mean(signal_valid)
                abs_sum = float(np.sum(np.abs(signal_demeaned)))
                if abs_sum < 1e-8:
                    continue
                positions = signal_demeaned * (target_gross / abs_sum)
            else:  # power
                ranks = sp_stats.rankdata(signal_valid)
                ranks_demeaned = ranks - np.mean(ranks)
                abs_sum = float(np.sum(np.abs(ranks_demeaned)))
                if abs_sum < 1e-8:
                    continue
                positions = ranks_demeaned * (target_gross / abs_sum)

            pnl = float(np.sum(positions * returns_valid))
            portfolio_return = pnl / target_gross

            daily_returns.append({
                'date': date_int,
                'portfolio_return': portfolio_return,
                'n_stocks': n_valid,
                'gross': target_gross,
            })
        return daily_returns

    n_workers = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_compute_one, available_intvs))

    all_returns_dfs = [pd.DataFrame(r) for r in results if r]
    if not all_returns_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_returns_dfs, ignore_index=True)
    avg = combined.groupby('date').agg({
        'portfolio_return': 'mean',
        'n_stocks': 'mean',
        'gross': 'first',
    }).reset_index()

    return avg


def compute_portfolio_pnl_by_subuniverse(
    signal_files_by_intv,
    subuniverse_keys,
    data_root,
    position_method='signal_weighted',
    target_gross=20e6,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily PnL contribution per sub-universe.

    Positions are computed using the **whole universe** signal (demean + scale
    to target_gross). The total daily PnL = sum(positions * returns) is then
    decomposed into contributions from each sub-universe by restricting the
    sum to stocks in that sub-universe.

    The result is averaged across all available intv values (same convention
    as ``compute_portfolio_returns_all_intv``).

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        subuniverse_keys: list of sub-universe keys whose PnL contributions
            should be computed. Each is loaded via ``load_universe_mask``.
        data_root: root of data directory.
        position_method: 'signal_weighted' (default) or 'power'.
        target_gross: target gross exposure.
        delay: number of intervals after signal to trade. Default 1.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT).
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with one row per (date, subuniverse) pair and columns:
            ['date', 'subuniverse', 'pnl_contribution']
        If nothing can be computed, returns an empty DataFrame.
    """
    available_intvs = sorted(signal_files_by_intv.keys())
    if not available_intvs:
        return pd.DataFrame()

    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)
    whole_mask = load_universe_mask('whole', data_root)

    # Load sub-universe masks once.
    sub_masks = {
        key: load_universe_mask(key, data_root) for key in subuniverse_keys
    }

    # Precompute ALL returns at once (vectorized over tidx). Cached at module
    # level so subsequent calls / universes reuse the same loaded data.
    all_returns = _precompute_all_returns(data_root, return_source)
    n_days = len(dates_arr)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, all_returns
    ) if exclude_limit else (None, None, None)

    # Batch-load signal matrices for all intvs in parallel (cached at module
    # level, so subsequent universes reuse the same loaded data).
    def _load_one_intv(intv):
        best_intv, signal_files = _select_signal_files_by_intv(
            signal_files_by_intv, intv
        )
        if best_intv is None:
            return None
        return intv, _load_signal_matrix(
            signal_files, sorted_uids, sort_idx, dates_arr
        )

    n_workers_load = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers_load) as ex:
        load_results = list(ex.map(_load_one_intv, available_intvs))

    signal_matrices = {}
    for r in load_results:
        if r is not None:
            signal_matrices[r[0]] = r[1]

    def _compute_one(intv):
        if intv not in signal_matrices:
            return None
        signal_matrix = signal_matrices[intv]
        tidx = intv + delay
        if tidx >= all_returns.shape[2]:
            return None

        rows = []
        for day_idx in range(n_days - 1):
            if not np.isfinite(signal_matrix[day_idx]).any():
                continue
            date_int = int(dates_arr[day_idx])

            avg_signal = signal_matrix[day_idx]
            returns = all_returns[day_idx + 1, :, tidx]

            if exclude_limit:
                limit_mask = _limit_hit_mask(
                    close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
                )
                returns = np.where(limit_mask, np.nan, returns)

            membership = whole_mask[day_idx]
            valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
            n_valid = int(valid_mask.sum())
            if n_valid < 10:
                continue

            signal_valid = avg_signal[valid_mask]
            returns_valid = returns[valid_mask]

            if position_method == 'signal_weighted':
                signal_demeaned = signal_valid - np.mean(signal_valid)
                abs_sum = float(np.sum(np.abs(signal_demeaned)))
                if abs_sum < 1e-8:
                    continue
                positions_valid = signal_demeaned * (target_gross / abs_sum)
            else:  # power
                ranks = sp_stats.rankdata(signal_valid)
                ranks_demeaned = ranks - np.mean(ranks)
                abs_sum = float(np.sum(np.abs(ranks_demeaned)))
                if abs_sum < 1e-8:
                    continue
                positions_valid = ranks_demeaned * (target_gross / abs_sum)

            # Decompose PnL by sub-universe membership. ``positions_valid`` is
            # indexed by ``valid_mask``; we compress each sub-universe mask
            # the same way so the elementwise product lines up.
            for sub_key in subuniverse_keys:
                sub_membership = sub_masks[sub_key][day_idx]
                sub_valid = (sub_membership == 1) & valid_mask
                if not sub_valid.any():
                    rows.append({
                        'date': date_int,
                        'subuniverse': sub_key,
                        'pnl_contribution': 0.0,
                    })
                    continue
                positions_sub = positions_valid[sub_valid[valid_mask]]
                returns_sub = returns_valid[sub_valid[valid_mask]]
                partial_pnl = float(np.sum(positions_sub * returns_sub))
                rows.append({
                    'date': date_int,
                    'subuniverse': sub_key,
                    'pnl_contribution': partial_pnl,
                })
        return rows

    n_workers = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_compute_one, available_intvs))

    all_rows = [r for r in results if r]
    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat([pd.DataFrame(r) for r in all_rows], ignore_index=True)
    # Average PnL contribution across intv values for each (date, subuniverse).
    avg = combined.groupby(['date', 'subuniverse']).agg({
        'pnl_contribution': 'mean',
    }).reset_index()

    return avg


def compute_portfolio_position_share_by_subuniverse(
    signal_files_by_intv,
    subuniverse_keys,
    data_root,
    position_method='signal_weighted',
    target_gross=20e6,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily position share per sub-universe.

    Positions are computed using the **whole universe** signal (demean + scale
    to target_gross, so ``sum(|positions|) == target_gross``). For each
    sub-universe, the daily share is::

        share_S = sum(|positions[mask_S]|) / target_gross

    The result is averaged across all available intv values (same convention
    as ``compute_portfolio_returns_all_intv``).

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        subuniverse_keys: list of sub-universe keys whose position shares
            should be computed. Each is loaded via ``load_universe_mask``.
        data_root: root of data directory.
        position_method: 'signal_weighted' (default) or 'power'.
        target_gross: target gross exposure.
        delay: number of intervals after signal to trade. Default 1.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT).
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with one row per (date, subuniverse) pair and columns:
            ['date', 'subuniverse', 'position_share']
        If nothing can be computed, returns an empty DataFrame.
    """
    available_intvs = sorted(signal_files_by_intv.keys())
    if not available_intvs:
        return pd.DataFrame()

    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)
    whole_mask = load_universe_mask('whole', data_root)

    sub_masks = {
        key: load_universe_mask(key, data_root) for key in subuniverse_keys
    }

    all_returns = _precompute_all_returns(data_root, return_source)
    n_days = len(dates_arr)

    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, all_returns
    ) if exclude_limit else (None, None, None)

    def _load_one_intv(intv):
        best_intv, signal_files = _select_signal_files_by_intv(
            signal_files_by_intv, intv
        )
        if best_intv is None:
            return None
        return intv, _load_signal_matrix(
            signal_files, sorted_uids, sort_idx, dates_arr
        )

    n_workers_load = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers_load) as ex:
        load_results = list(ex.map(_load_one_intv, available_intvs))

    signal_matrices = {}
    for r in load_results:
        if r is not None:
            signal_matrices[r[0]] = r[1]

    def _compute_one(intv):
        if intv not in signal_matrices:
            return None
        signal_matrix = signal_matrices[intv]
        tidx = intv + delay
        if tidx >= all_returns.shape[2]:
            return None

        rows = []
        for day_idx in range(n_days - 1):
            if not np.isfinite(signal_matrix[day_idx]).any():
                continue
            date_int = int(dates_arr[day_idx])

            avg_signal = signal_matrix[day_idx]
            returns = all_returns[day_idx + 1, :, tidx]

            if exclude_limit:
                limit_mask = _limit_hit_mask(
                    close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
                )
                returns = np.where(limit_mask, np.nan, returns)

            membership = whole_mask[day_idx]
            valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
            n_valid = int(valid_mask.sum())
            if n_valid < 10:
                continue

            signal_valid = avg_signal[valid_mask]
            returns_valid = returns[valid_mask]

            if position_method == 'signal_weighted':
                signal_demeaned = signal_valid - np.mean(signal_valid)
                abs_sum = float(np.sum(np.abs(signal_demeaned)))
                if abs_sum < 1e-8:
                    continue
                positions_valid = signal_demeaned * (target_gross / abs_sum)
            else:  # power
                ranks = sp_stats.rankdata(signal_valid)
                ranks_demeaned = ranks - np.mean(ranks)
                abs_sum = float(np.sum(np.abs(ranks_demeaned)))
                if abs_sum < 1e-8:
                    continue
                positions_valid = ranks_demeaned * (target_gross / abs_sum)

            # Position share for sub-universe S = sum(|positions[mask_S]|) /
            # target_gross. ``positions_valid`` is indexed by ``valid_mask``;
            # we compress each sub-universe mask the same way so the
            # elementwise product lines up.
            for sub_key in subuniverse_keys:
                sub_membership = sub_masks[sub_key][day_idx]
                sub_valid = (sub_membership == 1) & valid_mask
                if not sub_valid.any():
                    rows.append({
                        'date': date_int,
                        'subuniverse': sub_key,
                        'position_share': 0.0,
                    })
                    continue
                positions_sub = positions_valid[sub_valid[valid_mask]]
                gross_sub = float(np.sum(np.abs(positions_sub)))
                rows.append({
                    'date': date_int,
                    'subuniverse': sub_key,
                    'position_share': gross_sub / target_gross,
                })
        return rows

    n_workers = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_compute_one, available_intvs))

    all_rows = [r for r in results if r]
    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat([pd.DataFrame(r) for r in all_rows], ignore_index=True)
    # Average position share across intv values for each (date, subuniverse).
    avg = combined.groupby(['date', 'subuniverse']).agg({
        'position_share': 'mean',
    }).reset_index()

    return avg


# ── cumulative returns ────────────────────────────────────────────────────────

def compute_cumulative_returns(daily_returns_df):
    """Compute cumulative return curve from daily returns.

    Since gross exposure is held constant at target_gross each day (positions
    are re-sized to target_gross every day, not compounded), cumulative return
    is the sum of daily returns, not the compounded product.

    Args:
        daily_returns_df: pd.DataFrame with 'portfolio_return' column.

    Returns:
        pd.Series of cumulative returns (starting from 0).
    """
    if daily_returns_df.empty:
        return pd.Series(dtype=float)

    cum = daily_returns_df['portfolio_return'].cumsum()
    return cum


# ── monthly returns ───────────────────────────────────────────────────────────

def compute_monthly_returns(daily_returns_df):
    """Aggregate daily returns to monthly returns.

    Since gross exposure is held constant (re-sized to target_gross daily),
    monthly return is the sum of daily returns within the month.

    Args:
        daily_returns_df: pd.DataFrame with 'date' and 'portfolio_return' columns.

    Returns:
        pd.DataFrame with columns ['month', 'month_label', 'monthly_return', 'n_days'].
    """
    if daily_returns_df.empty:
        return pd.DataFrame(columns=['month', 'month_label',
                                     'monthly_return', 'n_days'])

    df = daily_returns_df.copy()
    df['month'] = df['date'].apply(lambda d: month_key(d))
    df = df.sort_values('date').reset_index(drop=True)

    monthly = []
    for mk, group in df.groupby('month'):
        # Sum daily returns within the month (constant gross exposure)
        monthly_return = float(group['portfolio_return'].sum())
        monthly.append({
            'month': mk,
            'month_label': month_label(mk),
            'monthly_return': monthly_return,
            'n_days': len(group),
        })

    return pd.DataFrame(monthly)


# ── signal statistics ─────────────────────────────────────────────────────────

def compute_signal_statistics(
    signal_files_by_intv,
    universe_key,
    data_root,
    intv=0,
):
    """Compute monthly statistics of factor signals on a given universe.

    Statistics: mean, std, skewness, kurtosis.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        intv: the interval index to use. Default 0.

    Returns:
        pd.DataFrame with columns:
            ['month_label', 'mean', 'std', 'skew', 'kurtosis', 'n_days', 'n_stocks']
    """
    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)

    uni_mask = load_universe_mask(universe_key, data_root)

    # Select signal files using fallback logic
    best_intv, signal_files = _select_signal_files_by_intv(
        signal_files_by_intv, intv
    )
    if best_intv is None:
        return pd.DataFrame()

    by_date = _group_files_by_date(signal_files)

    daily_stats = []
    for date_int in sorted(by_date.keys()):
        day_idx = date_to_index(date_int, dates_arr)
        if day_idx < 0:
            continue

        aligned_signals = []
        for fp in by_date[date_int]:
            aligned = _load_signal_aligned_vec(fp, sorted_uids, sort_idx)
            aligned_signals.append(aligned)

        if not aligned_signals:
            continue

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', category=RuntimeWarning)
            stacked = np.stack(aligned_signals, axis=0)
            if stacked.size == 0:
                continue
            avg_signal = np.nanmean(stacked, axis=0)

        membership = uni_mask[day_idx]
        valid_mask = (membership == 1) & np.isfinite(avg_signal)
        valid_signals = avg_signal[valid_mask]

        if len(valid_signals) < 10:
            continue

        daily_stats.append({
            'date': date_int,
            'mean': float(np.mean(valid_signals)),
            'std': float(np.std(valid_signals, ddof=1)),
            'skew': float(sp_stats.skew(valid_signals)),
            'kurtosis': float(sp_stats.kurtosis(valid_signals)),
            'n_stocks': int(len(valid_signals)),
        })

    df = pd.DataFrame(daily_stats)
    if df.empty:
        return pd.DataFrame()

    df['month'] = df['date'].apply(lambda d: month_key(d))

    # Aggregate to monthly
    monthly = df.groupby('month').agg({
        'mean': 'mean',
        'std': 'mean',
        'skew': 'mean',
        'kurtosis': 'mean',
        'date': 'count',
        'n_stocks': 'mean',
    }).reset_index()

    monthly.columns = ['month', 'mean', 'std', 'skew', 'kurtosis',
                       'n_days', 'n_stocks']
    monthly['month_label'] = monthly['month'].apply(month_label)

    # Put month_label first, drop the int 'month' column per user request
    monthly = monthly[['month_label', 'mean', 'std', 'skew', 'kurtosis',
                       'n_days', 'n_stocks']]
    return monthly


# ── IC (information coefficient) ──────────────────────────────────────────────

def compute_daily_ic(
    signal_files_by_intv,
    universe_key,
    data_root,
    intv=0,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily rank IC between factor signal and next-day return.

    Uses the original per-day approach: iterate over each trading day, load
    the signal files for that day, align to universe, compute next-day returns,
    and compute Spearman rank IC.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        intv: the interval index to use. Default 0.
        delay: number of intervals after signal to trade. Default 1.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['date', 'ic', 'n_stocks'].
    """
    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)

    uni_mask = load_universe_mask(universe_key, data_root)
    returns_data, adj_factor = _load_returns(data_root, return_source)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, returns_data
    ) if exclude_limit else (None, None, None)

    # Select signal files using fallback logic
    best_intv, signal_files = _select_signal_files_by_intv(
        signal_files_by_intv, intv
    )
    if best_intv is None:
        return pd.DataFrame()

    by_date = _group_files_by_date(signal_files)
    tidx = intv + delay

    ic_records = []
    for date_int in sorted(by_date.keys()):
        day_idx = date_to_index(date_int, dates_arr)
        if day_idx < 0 or day_idx + 1 >= len(dates_arr):
            continue

        aligned_signals = []
        for fp in by_date[date_int]:
            aligned = _load_signal_aligned_vec(fp, sorted_uids, sort_idx)
            aligned_signals.append(aligned)

        if not aligned_signals:
            continue

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', category=RuntimeWarning)
            stacked = np.stack(aligned_signals, axis=0)
            if stacked.size == 0:
                continue
            avg_signal = np.nanmean(stacked, axis=0)

        # Get next-day returns for day_idx + 1
        returns = _get_next_day_returns(
            returns_data, adj_factor, day_idx + 1, tidx,
            return_source=return_source,
        )

        # Exclude limit-hit stocks at trading tidx on day_idx+1
        if exclude_limit:
            limit_mask = _limit_hit_mask(
                close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
            )
            returns = np.where(limit_mask, np.nan, returns)

        membership = uni_mask[day_idx]
        valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
        n_valid = int(valid_mask.sum())
        if n_valid < 10:
            continue

        signal_valid = avg_signal[valid_mask]
        returns_valid = returns[valid_mask]

        rho, _ = sp_stats.spearmanr(signal_valid, returns_valid)
        ic_records.append({
            'date': date_int,
            'ic': float(rho) if not np.isnan(rho) else None,
            'n_stocks': n_valid,
        })

    return pd.DataFrame(ic_records)


def compute_daily_ic_all_intv(
    signal_files_by_intv,
    universe_key,
    data_root,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute daily rank IC averaged across all available intv values.

    For each intv value, compute the daily IC. Then align by date and average
    the IC across intv values.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['date', 'ic', 'n_stocks'].
    """
    available_intvs = sorted(signal_files_by_intv.keys())
    if not available_intvs:
        return pd.DataFrame()

    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_hashes, sort_idx = _build_stock_lookup(uid_arr)
    uni_mask = load_universe_mask(universe_key, data_root)

    # Precompute ALL returns at once (vectorized over tidx)
    all_returns = _precompute_all_returns(data_root, return_source)
    n_days = len(dates_arr)

    # Load limit prices and close intervals for limit filtering
    close_for_limit, up_lim, dn_lim = _load_limit_filter(
        data_root, return_source, all_returns
    ) if exclude_limit else (None, None, None)

    # Batch-load signal matrices for all intvs (cached at module level,
    # so subsequent universes / intvs reuse the same loaded data).
    # Load all intvs in parallel: each intv has ~700 npz files; serial
    # loading of 47 intvs takes ~60s, parallel brings it down to ~10s.
    def _load_one_intv(intv):
        best_intv, signal_files = _select_signal_files_by_intv(
            signal_files_by_intv, intv
        )
        if best_intv is None:
            return None
        return intv, _load_signal_matrix(
            signal_files, sorted_hashes, sort_idx, dates_arr
        )

    n_workers_load = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers_load) as ex:
        load_results = list(ex.map(_load_one_intv, available_intvs))

    signal_matrices = {}
    for r in load_results:
        if r is not None:
            signal_matrices[r[0]] = r[1]

    # Process each intv in parallel — the per-day rank IC computation is
    # CPU-bound numpy work that releases the GIL, so a thread pool gives
    # near-linear speedup across the 47 intv variants of factor B.
    def _compute_one_intv(intv):
        if intv not in signal_matrices:
            return None
        signal_matrix = signal_matrices[intv]
        tidx = intv + delay
        if tidx >= all_returns.shape[2]:
            return None

        ic_records = []
        for day_idx in range(n_days - 1):
            if not np.isfinite(signal_matrix[day_idx]).any():
                continue
            date_int = int(dates_arr[day_idx])

            avg_signal = signal_matrix[day_idx]
            returns = all_returns[day_idx + 1, :, tidx]

            if exclude_limit:
                limit_mask = _limit_hit_mask(
                    close_for_limit, up_lim, dn_lim, day_idx + 1, tidx
                )
                returns = np.where(limit_mask, np.nan, returns)

            membership = uni_mask[day_idx]
            valid_mask = (membership == 1) & np.isfinite(avg_signal) & np.isfinite(returns)
            n_valid = int(valid_mask.sum())
            if n_valid < 10:
                continue

            signal_valid = avg_signal[valid_mask]
            returns_valid = returns[valid_mask]

            ic = _rank_ic_fast(signal_valid, returns_valid)
            ic_records.append({
                'date': date_int,
                'ic': ic,
                'n_stocks': n_valid,
            })
        return ic_records

    n_workers = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_compute_one_intv, available_intvs))

    all_ics = [pd.DataFrame(r) for r in results if r]
    if not all_ics:
        return pd.DataFrame()

    combined = pd.concat(all_ics, ignore_index=True)
    avg = combined.groupby('date').agg({
        'ic': 'mean',
        'n_stocks': 'mean',
    }).reset_index()

    return avg


def compute_signal_statistics_all_intv(
    signal_files_by_intv,
    universe_key,
    data_root,
):
    """Compute monthly signal statistics averaged across all available intv values.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.

    Returns:
        pd.DataFrame with monthly statistics columns.
    """
    available_intvs = sorted(signal_files_by_intv.keys())
    if not available_intvs:
        return pd.DataFrame()

    uid_arr = np.array(load_uid(data_root))
    dates_arr = np.array(load_dates(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)
    uni_mask = load_universe_mask(universe_key, data_root)
    n_days = len(dates_arr)

    # Batch-load signal matrices for all intvs in parallel (cached at module
    # level, so subsequent universes reuse the same loaded data).
    def _load_one_intv(intv):
        best_intv, signal_files = _select_signal_files_by_intv(
            signal_files_by_intv, intv
        )
        if best_intv is None:
            return None
        return intv, _load_signal_matrix(
            signal_files, sorted_uids, sort_idx, dates_arr
        )

    n_workers_load = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers_load) as ex:
        load_results = list(ex.map(_load_one_intv, available_intvs))

    signal_matrices = {}
    for r in load_results:
        if r is not None:
            signal_matrices[r[0]] = r[1]

    # Process each intv in parallel — statistics computation is CPU-bound
    # numpy work that releases the GIL, so a thread pool gives near-linear
    # speedup across the 47 intv variants of factor B.
    def _compute_one(intv):
        if intv not in signal_matrices:
            return None
        signal_matrix = signal_matrices[intv]

        daily_stats = []
        for day_idx in range(n_days):
            if not np.isfinite(signal_matrix[day_idx]).any():
                continue
            membership = uni_mask[day_idx]
            valid_mask = (membership == 1) & np.isfinite(signal_matrix[day_idx])
            valid_signals = signal_matrix[day_idx][valid_mask]
            if len(valid_signals) < 10:
                continue
            # Fast inline skew/kurtosis (matches scipy defaults, ~10x faster)
            n = len(valid_signals)
            mean = float(np.mean(valid_signals))
            diff = valid_signals - mean
            m2 = float(np.sum(diff * diff)) / n
            m3 = float(np.sum(diff ** 3)) / n
            m4 = float(np.sum(diff ** 4)) / n
            if m2 > 0:
                m2_sqrt = np.sqrt(m2)
                sk = m3 / (m2_sqrt ** 3)
                ku = m4 / (m2 * m2) - 3
            else:
                sk = 0.0
                ku = 0.0
            daily_stats.append({
                'date': int(dates_arr[day_idx]),
                'mean': mean,
                'std': float(np.std(valid_signals, ddof=1)),
                'skew': float(sk),
                'kurtosis': float(ku),
                'n_stocks': n,
            })
        if not daily_stats:
            return None

        df = pd.DataFrame(daily_stats)
        df['month'] = df['date'].apply(lambda d: month_key(d))
        monthly = df.groupby('month').agg({
            'mean': 'mean',
            'std': 'mean',
            'skew': 'mean',
            'kurtosis': 'mean',
            'date': 'count',
            'n_stocks': 'mean',
        }).reset_index()
        monthly.columns = ['month', 'mean', 'std', 'skew', 'kurtosis',
                           'n_days', 'n_stocks']
        monthly['month_label'] = monthly['month'].apply(month_label)
        return monthly

    n_workers = min(len(available_intvs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_compute_one, available_intvs))

    all_stats = [df for df in results if df is not None]
    if not all_stats:
        return pd.DataFrame()

    combined = pd.concat(all_stats, ignore_index=True)
    avg = combined.groupby('month_label').agg({
        'mean': 'mean',
        'std': 'mean',
        'skew': 'mean',
        'kurtosis': 'mean',
        'n_days': 'mean',
        'n_stocks': 'mean',
    }).reset_index()
    avg = avg[['month_label', 'mean', 'std', 'skew', 'kurtosis',
               'n_days', 'n_stocks']]

    return avg


# ── monthly IC ────────────────────────────────────────────────────────────────

def compute_monthly_ic(
    signal_files_by_intv,
    universe_key,
    data_root,
    intv=0,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute monthly average of daily rank IC.

    For each day, compute the rank IC between the factor signal and the
    next-day return. Then aggregate to monthly averages.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        intv: the interval index to use. Default 0.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['month', 'month_label', 'monthly_ic',
        'n_days', 'n_stocks'].
    """
    daily_ic_df = compute_daily_ic(
        signal_files_by_intv, universe_key, data_root,
        intv=intv, delay=delay, return_source=return_source,
        exclude_limit=exclude_limit,
    )
    if daily_ic_df.empty:
        return pd.DataFrame()

    df = daily_ic_df.copy()
    df = df.dropna(subset=['ic'])
    if df.empty:
        return pd.DataFrame()

    df['month'] = df['date'].apply(lambda d: month_key(d))
    df = df.sort_values('date').reset_index(drop=True)

    monthly = []
    for mk, group in df.groupby('month'):
        monthly_ic = float(group['ic'].mean())
        monthly.append({
            'month': mk,
            'month_label': month_label(mk),
            'monthly_ic': monthly_ic,
            'n_days': len(group),
            'n_stocks': float(group['n_stocks'].mean()) if 'n_stocks' in group.columns else 0,
        })

    return pd.DataFrame(monthly)


def compute_monthly_ic_all_intv(
    signal_files_by_intv,
    universe_key,
    data_root,
    delay=1,
    return_source='raw',
    exclude_limit=False,
):
    """Compute monthly IC averaged across all available intv values.

    For each intv value, compute the daily IC, aggregate to monthly averages.
    Then average the monthly IC across intv values.

    Args:
        signal_files_by_intv: dict {intv_value: [(date, path), ...]}.
        universe_key: universe to filter by.
        data_root: root of data directory.
        return_source: 'raw' (close-based) or 'dsrt' (pre-computed DSRT). Default 'raw'.
        exclude_limit: if True, mask out stocks that hit limit-up/down at the
            trading tidx on day_idx+1. Default False.

    Returns:
        pd.DataFrame with columns ['month', 'month_label', 'monthly_ic',
        'n_days', 'n_stocks'].
    """
    available_intvs = sorted(signal_files_by_intv.keys())
    if not available_intvs:
        return pd.DataFrame()

    # Reuse the already-optimized compute_daily_ic_all_intv (which uses the
    # module-level signal-matrix cache and parallel intv processing), then
    # aggregate to monthly. This avoids 47x redundant npz loading that the
    # previous per-intv compute_monthly_ic approach incurred.
    daily_ic_df = compute_daily_ic_all_intv(
        signal_files_by_intv, universe_key, data_root,
        delay=delay, return_source=return_source,
        exclude_limit=exclude_limit,
    )
    if daily_ic_df.empty:
        return pd.DataFrame()

    df = daily_ic_df.copy()
    df = df.dropna(subset=['ic'])
    if df.empty:
        return pd.DataFrame()

    df['month'] = df['date'].apply(lambda d: month_key(d))
    df = df.sort_values('date').reset_index(drop=True)

    monthly = []
    for mk, group in df.groupby('month'):
        monthly_ic = float(group['ic'].mean())
        monthly.append({
            'month': mk,
            'month_label': month_label(mk),
            'monthly_ic': monthly_ic,
            'n_days': len(group),
            'n_stocks': float(group['n_stocks'].mean()) if 'n_stocks' in group.columns else 0,
        })

    return pd.DataFrame(monthly)
