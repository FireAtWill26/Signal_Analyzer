"""data_loader.py — mmap-based readers for universe / price / signal data.

Layout assumptions (verified against /dfs/dataset/230-1724661625521/data/dataprod):

  __universe/uid.N128C
      dtype <U32, shape (5459,) — whole-universe stock codes.
  __universe/dates.NI
      dtype <i4, shape (1511,) — trading dates as YYYYMMDD ints.
  <UNIV>/<UNIV>.N,5860c   (e.g. ZZ500/ZZ500.N,5860c)
      dtype <i1, shape (1698, 5860) — daily membership mask (1=member).
      Only the first 1511 rows correspond to dates in dates.NI.
      "whole" universe is special: every stock in uid.N128C is a member.
  ForwardAdjPrices/ForwardAdjPrices.S_DQ_CLOSE.N,5860f
      dtype <f4, shape (1698, 5860) — daily close (raw, 不复权).
  ForwardAdjPrices/ForwardAdjPrices.S_DQ_ADJCLOSE.N,5860f
      dtype <f4, shape (1698, 5860) — daily close (前复权).
  ForwardAdjPrices/ForwardAdjPrices.S_DQ_ADJCLOSE_BACKWARD.N,5860f
      dtype <f4, shape (1698, 5860) — daily close (后复权).
  ForwardAdjPrices/ForwardAdjPrices.S_DQ_ADJFACTOR.N,5860f
      dtype <f4, shape (1698, 5860) — daily adj factor (复权因子).
  IntervalFull/IntervalFull.close.N49,5860f
      dtype <f4, shape (1698, 49, 5860) — interval close (49 个 5min 区间).
      按 compute_return.py 约定，访问时视为 (n_days, n_stocks, n_intv)，
      因此本模块对外暴露 transpose 后的 (n_days, n_stocks, n_intv)。
  BlockN/block4n.N,5860I
      dtype <i4, shape (1698, 5860) — 板块归属 (0=上证, 1=深证, 2=创业板, 3=科创板).
  ZZ3000Index/ZZ3000.weight.N,5860f
      dtype <f4, shape (1698, 5860) — ZZ3000 成分股权重 (基准用).

  Signal files (npz):
      example_signal/YYYY/MM/DD/alpha.YYYYMMDD.intv49i<XX>.npz
      Keys: alphaV1, alphaV2, alphaV3, vValidTag, vStockCode, vDumpInfo
      Multiple intv variants per day (i00/i06/i12/i24/i30/i36)
"""

import os
import glob
import json
import numpy as np


# ── constants ─────────────────────────────────────────────────────────────────

N_DAYS_CAPACITY = 1698
N_STOCKS = 5860
N_VALID_DAYS = 1511  # dates.NI length
N_INTERVALS = 49     # IntervalFull interval count


# ── config ────────────────────────────────────────────────────────────────────

def load_config(config_path='config.json'):
    """Load analyzer config."""
    with open(config_path, 'r') as f:
        return json.load(f)


# ── universe / dates ──────────────────────────────────────────────────────────

# Module-level cache: avoid re-copying multi-GB arrays on every call.
# Keyed by (data_root, optional universe_key) so different configs don't collide.
_DATA_CACHE = {}


def load_uid(data_root):
    """Load whole-universe stock codes. dtype <U32."""
    ck = ('uid', data_root)
    if ck not in _DATA_CACHE:
        path = os.path.join(data_root, '__universe/uid.N128C')
        arr = np.memmap(path, dtype='<U32', mode='r')
        _DATA_CACHE[ck] = np.array(arr)
    return _DATA_CACHE[ck]


def load_dates(data_root):
    """Load trading dates as YYYYMMDD int32 array.

    Truncated to ``N_VALID_DAYS`` to match price/adj_factor arrays, since
    the dates.NI file may contain more entries than the valid price rows.
    """
    ck = ('dates', data_root)
    if ck not in _DATA_CACHE:
        path = os.path.join(data_root, '__universe/dates.NI')
        arr = np.memmap(path, dtype='<i4', mode='r')
        _DATA_CACHE[ck] = np.array(arr[:N_VALID_DAYS])
    return _DATA_CACHE[ck]


def load_universe_mask(universe_key, data_root):
    """Load daily membership mask for a universe.

    Returns:
        np.ndarray (n_days, n_stocks) of int8 (0/1).
        Shape is (N_VALID_DAYS, N_STOCKS) = (1511, 5860).
    """
    ck = ('universe_mask', universe_key, data_root)
    if ck not in _DATA_CACHE:
        config = load_config()
        universes = {u['key']: u for u in config['universe_list']}
        if universe_key not in universes:
            raise ValueError(f"Unknown universe: {universe_key}")

        uni = universes[universe_key]
        if uni['type'] == 'uid':
            mask = np.ones((N_VALID_DAYS, N_STOCKS), dtype=np.int8)
        elif uni['type'] == 'block':
            block_id = uni['block_id']
            block = _load_block4n(data_root)
            mask = (block == block_id).astype(np.int8)
        elif uni['type'] == 'complement':
            # Start from the whole universe, then subtract each listed
            # universe's membership. ``base`` defaults to 'whole'.
            base_key = uni.get('base', 'whole')
            mask = load_universe_mask(base_key, data_root).copy()
            for sub_key in uni.get('subtract', []):
                sub_mask = load_universe_mask(sub_key, data_root)
                mask = np.where(sub_mask == 1, 0, mask)
        else:
            rel_path = uni['path']
            full_path = os.path.join(data_root, rel_path)
            arr = np.memmap(full_path, dtype='<i1', mode='r').reshape(-1, N_STOCKS)
            mask = np.array(arr[:N_VALID_DAYS, :])
        _DATA_CACHE[ck] = mask
    return _DATA_CACHE[ck]


def _load_block4n(data_root):
    """Load BlockN/block4n.N,5860I as int32 (n_days, n_stocks)."""
    ck = ('block4n', data_root)
    if ck not in _DATA_CACHE:
        path = os.path.join(data_root, 'BlockN/block4n.N,5860I')
        arr = np.memmap(path, dtype='<i4', mode='r').reshape(-1, N_STOCKS)
        _DATA_CACHE[ck] = np.array(arr[:N_VALID_DAYS, :])
    return _DATA_CACHE[ck]


# ── prices ────────────────────────────────────────────────────────────────────

def load_close_prices(data_root, adjust='forward'):
    """Load daily close prices.

    Args:
        adjust: 'raw' (不复权), 'forward' (前复权), 'backward' (后复权).

    Returns:
        np.ndarray (N_VALID_DAYS, N_STOCKS) of float32.
    """
    ck = ('close_prices', data_root, adjust)
    if ck not in _DATA_CACHE:
        config = load_config()
        price_files = config['price_files']
        if adjust not in price_files:
            raise ValueError(f"Unknown adjust type: {adjust}. "
                             f"Must be one of {list(price_files.keys())}")

        rel_path = price_files[adjust]
        full_path = os.path.join(data_root, rel_path)
        arr = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS)
        _DATA_CACHE[ck] = np.array(arr[:N_VALID_DAYS, :])
    return _DATA_CACHE[ck]


def load_interval_close(data_root):
    """Load IntervalFull.close.N49,5860f as (n_days, n_stocks, n_intv) float32.

    File layout is (n_days, n_stocks, n_intervals) where the last dim is 49.
    """
    ck = ('interval_close', data_root)
    if ck not in _DATA_CACHE:
        config = load_config()
        rel_path = config['interval_close']
        full_path = os.path.join(data_root, rel_path)
        arr = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS, N_INTERVALS)
        _DATA_CACHE[ck] = np.array(arr[:N_VALID_DAYS, :, :])
    return _DATA_CACHE[ck]


def load_adj_factor(data_root):
    """Load CAX/adjfactor.N,5860f as (n_days, n_stocks) float32.

    This is the forward-adjustment factor: values are 1.0 on normal days
    and differ from 1.0 only on ex-dividend / ex-split dates. Used to
    forward-adjust the previous-day close when computing next-day returns
    (see compute_return.py).
    """
    ck = ('adj_factor', data_root)
    if ck not in _DATA_CACHE:
        config = load_config()
        rel_path = config['adj_factor']
        full_path = os.path.join(data_root, rel_path)
        arr = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS)
        _DATA_CACHE[ck] = np.array(arr[:N_VALID_DAYS, :])
    return _DATA_CACHE[ck]


def load_zz3000_weights(data_root):
    """Load ZZ3000Index/ZZ3000.weight.N,5860f as (n_days, n_stocks) float32.

    Used as the benchmark position weights for ZZ3000.
    """
    ck = ('zz3000_weights', data_root)
    if ck not in _DATA_CACHE:
        config = load_config()
        rel_path = config['zz3000_weight']
        full_path = os.path.join(data_root, rel_path)
        arr = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS)
        _DATA_CACHE[ck] = np.array(arr[:N_VALID_DAYS, :])
    return _DATA_CACHE[ck]


def load_dsrt_returns(data_root):
    """Load CNE5Ret/CNE5Ret.DSRT.d1.N49,5664f as (n_days, n_stocks, n_intv) float32.

    This file contains pre-computed daily DSRT (Daily Standardized Return)
    values. Each value is already a return ratio (not a price), so no
    adj-factor or close-price computation is needed.

    File shape: (1568, 5664, 49). The DSRT file has 5664 columns (vs 5860 in
    IntervalFull), but both map uid stocks to the first N columns. We pad with
    NaN to N_STOCKS columns to align with close-based returns.
    """
    ck = ('dsrt_returns', data_root)
    if ck not in _DATA_CACHE:
        config = load_config()
        rel_path = config['dsrt_return']
        full_path = os.path.join(data_root, rel_path)
        arr = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, 5664, N_INTERVALS)
        arr = np.array(arr[:N_VALID_DAYS, :, :])
        if arr.shape[1] < N_STOCKS:
            padded = np.full((arr.shape[0], N_STOCKS, arr.shape[2]), np.nan, dtype=np.float32)
            padded[:, :arr.shape[1], :] = arr
            _DATA_CACHE[ck] = padded
        else:
            _DATA_CACHE[ck] = arr[:, :N_STOCKS, :]
    return _DATA_CACHE[ck]


def load_limit_prices(data_root):
    """Load Limits/UpLimPrice.N,5860f and DnLimPrice.N,5860f.

    Returns:
        (up_lim, dn_lim) tuple of (N_VALID_DAYS, N_STOCKS) float32 arrays.
        Values are daily limit-up / limit-down prices per stock.
    """
    ck = ('limit_prices', data_root)
    if ck not in _DATA_CACHE:
        up_path = os.path.join(data_root, 'Limits/UpLimPrice.N,5860f')
        dn_path = os.path.join(data_root, 'Limits/DnLimPrice.N,5860f')
        up_arr = np.memmap(up_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS)
        dn_arr = np.memmap(dn_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS)
        _DATA_CACHE[ck] = (np.array(up_arr[:N_VALID_DAYS, :]), np.array(dn_arr[:N_VALID_DAYS, :]))
    return _DATA_CACHE[ck]


# ── signals ───────────────────────────────────────────────────────────────────

def scan_signal_files(signal_root):
    """Scan signal directory for npz files.

    Returns:
        dict: {intv_value: [(date_int, filepath), ...]} sorted by date.
        intv_value is an integer 0-49.
    """
    files = sorted(glob.glob(os.path.join(signal_root, '**', '*.npz'),
                             recursive=True))
    by_intv = {}
    for f in files:
        bn = os.path.basename(f)
        # alpha.YYYYMMDD.intv49i<XX>.npz
        parts = bn.split('.')
        if len(parts) < 3 or parts[0] != 'alpha':
            continue
        if not parts[1].isdigit():
            continue
        date_int = int(parts[1])
        intv_token = parts[2]  # intv49i00
        # Extract the intv value: the number after 'i' in the token
        # e.g., 'intv49i00' -> 0, 'intv49i36' -> 36
        if intv_token.startswith('intv'):
            rest = intv_token[4:]  # '49i00'
            if 'i' in rest:
                idx = rest.index('i')
                intv_str = rest[idx + 1:]  # '00'
                if not intv_str.isdigit():
                    continue
                intv_value = int(intv_str)
            else:
                continue  # Not a valid intv file
        else:
            continue  # Not a valid intv file

        by_intv.setdefault(intv_value, []).append((date_int, f))

    for tag in by_intv:
        by_intv[tag].sort(key=lambda x: x[0])
    return by_intv


def load_signal_npz(path):
    """Load a signal npz file, return dict with stocks, pred, valid_tag."""
    d = np.load(path)
    return {
        'alphaV1': d['alphaV1'].astype(np.float32),
        'alphaV2': d['alphaV2'].astype(np.float32) if 'alphaV2' in d.files else None,
        'alphaV3': d['alphaV3'].astype(np.float32) if 'alphaV3' in d.files else None,
        'vStockCode': d['vStockCode'],
        'vValidTag': d['vValidTag'].astype(bool) if 'vValidTag' in d.files else None,
        'vDumpInfo': d['vDumpInfo'] if 'vDumpInfo' in d.files else None,
    }


# ── date helpers ──────────────────────────────────────────────────────────────

def date_to_index(date_int, dates_arr):
    """Find index of date_int in dates_arr. Returns -1 if not found."""
    matches = np.where(dates_arr == date_int)[0]
    return int(matches[0]) if len(matches) > 0 else -1


def month_key(date_int):
    """Convert YYYYMMDD int to YYYYMM int."""
    return date_int // 100


def month_label(mk):
    """Format YYYYMM as 'YYYY-MM'."""
    return f"{mk // 100}-{mk % 100:02d}"


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = load_config()
    data_root = config['data_root']

    uid = load_uid(data_root)
    dates = load_dates(data_root)
    print(f"uid: len={len(uid)}, first 3: {uid[:3]}")
    print(f"dates: len={len(dates)}, first 3: {dates[:3]}")

    uni_mask = load_universe_mask('ZZ500', data_root)
    print(f"ZZ500 mask: shape={uni_mask.shape}, "
          f"day 0 members: {uni_mask[0].sum()}, "
          f"day 1000 members: {uni_mask[1000].sum()}")

    close = load_close_prices(data_root, adjust='forward')
    print(f"close (forward adj): shape={close.shape}, "
          f"day 0 first 3: {close[0, :3]}")

    by_intv = scan_signal_files(config['signal_root'])
    print(f"signal intv variants: {sorted(by_intv.keys())}")
    print(f"  i00 files: {len(by_intv.get('i00', []))}")
