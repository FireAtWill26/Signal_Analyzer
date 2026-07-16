"""Analyze why tidx=0 gives anomalously high returns."""
import numpy as np
from data_loader import load_interval_close, load_adj_factor, load_uid, load_universe_mask

data_root = "/dfs/dataset/230-1724661625521/data/dataprod"
close = load_interval_close(data_root)
adj = load_adj_factor(data_root)
uid = np.array(load_uid(data_root))

# Check the return at tidx=0 vs tidx=48 for the whole universe
# Return at tidx: close[day+1, tidx] / (close[day, tidx] * adj[day+1]) - 1

day = 100
whole_mask = load_universe_mask('whole', data_root)
members = whole_mask[day] == 1

print(f"Day {day}, whole universe members: {members.sum()}")

# Compute returns at different tidx
for tidx in [0, 1, 6, 12, 24, 36, 48]:
    base = close[day, :5860, tidx] * adj[day + 1]
    ret = close[day + 1, :5860, tidx] / base - 1
    valid = members & np.isfinite(ret)
    if valid.sum() > 0:
        print(f"  tidx={tidx}: n_valid={valid.sum()}, "
              f"mean_ret={np.nanmean(ret[valid]):.6f}, "
              f"std_ret={np.nanstd(ret[valid]):.6f}")

# Check: what is the correlation between the signal and the return at tidx=0?
# This would tell us if the factor is actually predictive at tidx=0

# Let's also check the raw return values for a few stocks at tidx=0
print(f"\nReturns at tidx=0 for first 10 stocks (day {day}):")
base0 = close[day, :5860, 0] * adj[day + 1]
ret0 = close[day + 1, :5860, 0] / base0 - 1
for s in range(10):
    print(f"  stock {uid[s]}: close[d,0]={close[day, s, 0]:.2f}, "
          f"close[d+1,0]={close[day+1, s, 0]:.2f}, "
          f"adj[d+1]={adj[day+1, s]:.4f}, "
          f"ret={ret0[s]:.6f}")

# Check returns at tidx=48
print(f"\nReturns at tidx=48 for first 10 stocks (day {day}):")
base48 = close[day, :5860, 48] * adj[day + 1]
ret48 = close[day + 1, :5860, 48] / base48 - 1
for s in range(10):
    print(f"  stock {uid[s]}: close[d,48]={close[day, s, 48]:.2f}, "
          f"close[d+1,48]={close[day+1, s, 48]:.2f}, "
          f"adj[d+1]={adj[day+1, s]:.4f}, "
          f"ret={ret48[s]:.6f}")
