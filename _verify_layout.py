"""Verify the data layout of IntervalFull.close by comparing tidx=48 with daily close."""
import os
import json
import numpy as np
from data_loader import (
    load_close_prices, load_uid, N_STOCKS, N_INTERVALS, N_VALID_DAYS,
)

data_root = "/dfs/dataset/230-1724661625521/data/dataprod"

# Load daily close (raw = S_DQ_CLOSE)
daily_close_raw = load_close_prices(data_root, adjust='raw')
print(f"daily_close_raw shape: {daily_close_raw.shape}")

# Load uid
uid = np.array(load_uid(data_root))

# Read the interval close file TWO ways:
# A) Current: reshape(-1, 5860, 49) — assumes (n_days, n_stocks, n_intv)
# B) Fixed: reshape(-1, 49, 5860) then transpose(0,2,1) — assumes (n_days, n_intv, n_stocks)

config_path = '/dfs/data/tools/analyzer/config.json'
import json
with open(config_path) as f:
    config = json.load(f)
rel_path = config['interval_close']
full_path = os.path.join(data_root, rel_path)

# Way A (current)
arr_a = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, N_STOCKS, N_INTERVALS)
arr_a = np.array(arr_a[:N_VALID_DAYS, :, :])

# Way B (fixed)
arr_b = np.memmap(full_path, dtype='<f4', mode='r').reshape(-1, N_INTERVALS, N_STOCKS)
arr_b = arr_b.transpose(0, 2, 1)
arr_b = np.array(arr_b[:N_VALID_DAYS, :, :])

# Check: does close[day, stock, 48] match daily_close_raw[day, stock]?
day = 100
print(f"\nDay {day}:")
print(f"{'stock':>15} {'daily_raw':>12} {'A[t48]':>12} {'B[t48]':>12} {'A_match':>8} {'B_match':>8}")
for s in [0, 100, 500, 1000, 2000, 3000, 4000, 5000]:
    daily = daily_close_raw[day, s]
    a48 = arr_a[day, s, 48]
    b48 = arr_b[day, s, 48]
    a_match = np.isclose(a48, daily, rtol=0.001) if np.isfinite(a48) and np.isfinite(daily) else False
    b_match = np.isclose(b48, daily, rtol=0.001) if np.isfinite(b48) and np.isfinite(daily) else False
    print(f"{uid[s]:>15} {daily:>12.2f} {a48:>12.2f} {b48:>12.2f} {str(a_match):>8} {str(b_match):>8}")

# Count matches across all stocks for day 100
a_matches = 0
b_matches = 0
total = 0
for s in range(N_STOCKS):
    daily = daily_close_raw[day, s]
    a48 = arr_a[day, s, 48]
    b48 = arr_b[day, s, 48]
    if np.isfinite(daily):
        total += 1
        if np.isfinite(a48) and np.isclose(a48, daily, rtol=0.001):
            a_matches += 1
        if np.isfinite(b48) and np.isclose(b48, daily, rtol=0.001):
            b_matches += 1

print(f"\nTotal stocks with valid daily close: {total}")
print(f"Way A (current reshape) matches: {a_matches}/{total}")
print(f"Way B (transpose) matches: {b_matches}/{total}")
