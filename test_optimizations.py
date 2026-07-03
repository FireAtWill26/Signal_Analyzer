"""Test script to verify optimizations in analyzer.py."""
import sys
import time
import numpy as np

sys.path.insert(0, '/dfs/data/tools/analyzer')

from analyzer import (
    _rank_ic_fast, _precompute_all_returns, _build_stock_lookup,
    _load_signal_aligned_vec, _load_signal_matrix,
    compute_daily_ic, compute_daily_ic_all_intv,
    _map_signal_to_universe,
)
from data_loader import (
    load_uid, load_dates, load_interval_close, load_adj_factor,
    scan_signal_files, load_config, load_signal_npz,
)


def test_rank_ic_fast():
    """Verify _rank_ic_fast matches scipy.stats.spearmanr."""
    from scipy import stats as sp_stats
    rng = np.random.default_rng(42)
    for _ in range(10):
        n = rng.integers(50, 500)
        signal = rng.standard_normal(n)
        returns = signal * 0.5 + rng.standard_normal(n) * 0.5
        rho_sp, _ = sp_stats.spearmanr(signal, returns)
        rho_fast = _rank_ic_fast(signal, returns)
        print(f"  n={n}: spearmanr={rho_sp:.6f}, fast={rho_fast:.6f}, "
              f"diff={abs(rho_sp - rho_fast):.2e}")
        assert abs(rho_sp - rho_fast) < 1e-6, f"Mismatch: {rho_sp} vs {rho_fast}"
    print("PASS: _rank_ic_fast matches spearmanr")


def test_int64_hash_mapping():
    """Verify vectorized mapping matches dict-based mapping."""
    cfg = load_config()
    data_root = cfg['data_root']
    uid_arr = np.array(load_uid(data_root))
    sorted_uids, sort_idx = _build_stock_lookup(uid_arr)
    stock_to_idx = {s: i for i, s in enumerate(uid_arr)}

    by_intv = scan_signal_files(cfg['signal_root'])
    files = by_intv.get(0, [])[:5]
    for date_int, fp in files:
        sig = load_signal_npz(fp)
        stocks, pred = sig['vStockCode'], sig['alphaV1']
        aligned_dict = _map_signal_to_universe(stocks, pred, stock_to_idx)
        aligned_vec = _load_signal_aligned_vec(fp, sorted_uids, sort_idx)
        diff = np.nanmax(np.abs(aligned_dict - aligned_vec))
        print(f"  date={date_int}: max diff = {diff:.2e}")
        assert diff < 1e-6, f"Mismatch: max diff {diff}"
    print("PASS: vectorized mapping matches dict-based mapping")


def test_precompute_all_returns():
    """Verify _precompute_all_returns matches per-day computation."""
    cfg = load_config()
    data_root = cfg['data_root']
    close = load_interval_close(data_root)
    adj = load_adj_factor(data_root)

    all_returns = _precompute_all_returns(data_root, 'raw')
    print(f"  all_returns shape: {all_returns.shape}")

    rng = np.random.default_rng(42)
    n_days = close.shape[0]
    n_stocks = close.shape[1]
    n_intv = close.shape[2]
    max_diff = 0
    for _ in range(100):
        d = rng.integers(1, n_days)
        s = rng.integers(0, n_stocks)
        t = rng.integers(0, n_intv)
        base = close[d - 1, s, t] * adj[d, s]
        expected = close[d, s, t] / base - 1
        actual = all_returns[d, s, t]
        if np.isfinite(expected) and np.isfinite(actual):
            max_diff = max(max_diff, abs(expected - actual))
    print(f"  max diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Mismatch: max diff {max_diff}"
    print("PASS: _precompute_all_returns matches per-day computation")


def test_compute_daily_ic_all_intv():
    """Test performance and correctness of compute_daily_ic_all_intv."""
    cfg = load_config()
    data_root = cfg['data_root']
    by_intv = scan_signal_files(cfg['signal_root'])
    print(f"  available intvs: {sorted(by_intv.keys())}")

    t0 = time.time()
    ic_all = compute_daily_ic_all_intv(
        by_intv, 'ZZ500', data_root,
        delay=1, return_source='raw', exclude_limit=False,
    )
    t1 = time.time()
    print(f"  compute_daily_ic_all_intv took {t1 - t0:.2f}s")
    print(f"  result shape: {ic_all.shape}")
    if not ic_all.empty:
        print(f"  mean IC: {ic_all['ic'].mean():.4f}")
        print(f"  IC IR: {ic_all['ic'].mean() / ic_all['ic'].std():.4f}")
    print("PASS: compute_daily_ic_all_intv completed")


if __name__ == '__main__':
    print("=== Testing _rank_ic_fast ===")
    test_rank_ic_fast()
    print()
    print("=== Testing vectorized mapping ===")
    test_int64_hash_mapping()
    print()
    print("=== Testing _precompute_all_returns ===")
    test_precompute_all_returns()
    print()
    print("=== Testing compute_daily_ic_all_intv ===")
    test_compute_daily_ic_all_intv()
