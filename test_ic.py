"""Debug script for IC computation issues."""
import sys
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, '/dfs/data/tools/analyzer')

from analyzer import (
    compute_daily_ic, compute_monthly_ic,
    compute_daily_ic_all_intv, compute_monthly_ic_all_intv,
)
from data_loader import scan_signal_files, load_config


def test_single_intv_ic():
    cfg = load_config()
    by_intv = scan_signal_files(cfg['signal_root'])
    print(f"available intvs: {sorted(by_intv.keys())}")

    try:
        ic_df = compute_daily_ic(
            by_intv, 'ZZ500', cfg['data_root'],
            intv=0, delay=1, return_source='raw',
            exclude_limit=False,
        )
        print(f"compute_daily_ic: shape={ic_df.shape}")
        print(ic_df.head())
        print(f"mean IC: {ic_df['ic'].mean() if not ic_df.empty else 'N/A'}")
    except Exception as e:
        print(f"ERROR in compute_daily_ic: {e}")
        traceback.print_exc()


def test_monthly_ic():
    cfg = load_config()
    by_intv = scan_signal_files(cfg['signal_root'])

    try:
        monthly_df = compute_monthly_ic(
            by_intv, 'ZZ500', cfg['data_root'],
            intv=0, delay=1, return_source='raw',
            exclude_limit=False,
        )
        print(f"compute_monthly_ic: shape={monthly_df.shape}")
        print(monthly_df.head() if not monthly_df.empty else "EMPTY")
    except Exception as e:
        print(f"ERROR in compute_monthly_ic: {e}")
        traceback.print_exc()


def test_all_intv_ic():
    cfg = load_config()
    by_intv = scan_signal_files(cfg['signal_root'])

    try:
        ic_df = compute_daily_ic_all_intv(
            by_intv, 'ZZ500', cfg['data_root'],
            delay=1, return_source='raw', exclude_limit=False,
        )
        print(f"compute_daily_ic_all_intv: shape={ic_df.shape}")
        print(ic_df.head() if not ic_df.empty else "EMPTY")
    except Exception as e:
        print(f"ERROR in compute_daily_ic_all_intv: {e}")
        traceback.print_exc()


def test_monthly_ic_all_intv():
    cfg = load_config()
    by_intv = scan_signal_files(cfg['signal_root'])

    try:
        monthly_df = compute_monthly_ic_all_intv(
            by_intv, 'ZZ500', cfg['data_root'],
            delay=1, return_source='raw', exclude_limit=False,
        )
        print(f"compute_monthly_ic_all_intv: shape={monthly_df.shape}")
        print(monthly_df.head() if not monthly_df.empty else "EMPTY")
    except Exception as e:
        print(f"ERROR in compute_monthly_ic_all_intv: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    print("=== Test single intv IC ===")
    test_single_intv_ic()
    print()
    print("=== Test monthly IC (single intv) ===")
    test_monthly_ic()
    print()
    print("=== Test all intv IC ===")
    test_all_intv_ic()
    print()
    print("=== Test monthly IC all intv ===")
    test_monthly_ic_all_intv()
