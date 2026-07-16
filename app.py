"""app.py — Streamlit UI for factor signal analysis.

Features:
  - Select factor signal files (multiple intv variants)
  - Optional second factor for side-by-side comparison
  - Select universe (whole/A500/ZZ100/ZZ500/ZZ800/ZZ1000/ZZ1500/ZZ1800/ZZ2000/
    ZZ3000/HS300/上证/深证/创业板/科创板)
  - Position methods: signal_weighted (default) / power (rank → demean → scale)
  - Cumulative return curve with ZZ3000 benchmark as comparison
  - Monthly returns table with top-5 highlight
  - Multi-universe monthly comparison table (whole / HS300 / 创业板 / 科创板)
  - Monthly signal statistics (mean/std/skew/kurtosis)
  - Daily rank IC curve

Usage:
  streamlit run /dfs/data/tools/analyzer/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the analyzer directory to path so we can import data_loader and analyzer
ANALYZER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ANALYZER_DIR)

from data_loader import (
    load_config, scan_signal_files, load_uid, load_dates,
    load_universe_mask, load_close_prices, load_signal_npz,
    date_to_index, month_key, month_label
)
from analyzer import (
    compute_portfolio_returns, compute_signal_weighted_returns,
    compute_power_returns, compute_zz3000_benchmark_returns,
    compute_monthly_returns, compute_cumulative_returns,
    compute_signal_statistics, compute_daily_ic,
    compute_monthly_ic,
    compute_portfolio_returns_all_intv,
    compute_portfolio_pnl_by_subuniverse,
    compute_portfolio_position_share_by_subuniverse,
    compute_top_stocks_by_month,
    compute_signal_statistics_all_intv,
    compute_daily_ic_all_intv,
    compute_monthly_ic_all_intv,
    _find_best_intv
)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="因子信号分析工具",
    page_icon="📊",
    layout="wide",
)


# ── load config ───────────────────────────────────────────────────────────────

@st.cache_data
def get_config():
    return load_config()


@st.cache_data
def get_universe_options():
    cfg = get_config()
    return {u['key']: u['name'] for u in cfg['universe_list']}


@st.cache_data
def get_signal_files(signal_root):
    """Scan signal directory, return dict {intv_value: [(date, path), ...]}."""
    return scan_signal_files(signal_root)


@st.cache_data
def get_date_range(signal_root):
    """Get min/max signal dates across all intv variants."""
    by_intv = scan_signal_files(signal_root)
    all_dates = []
    for _, files in by_intv.items():
        all_dates.extend([d for d, _ in files])
    if not all_dates:
        return None, None
    return min(all_dates), max(all_dates)


# ── cached analysis functions ─────────────────────────────────────────────────

def _to_hashable(by_intv):
    """Convert dict {intv: [(date, path), ...]} to hashable tuple."""
    return tuple((intv, tuple(files)) for intv, files in by_intv.items())


@st.cache_data(show_spinner="计算组合收益...")
def cached_compute_returns(
    signal_files_by_intv_tuple,
    universe_key,
    position_method,
    target_gross,
    intv,
    delay,
    return_source,
    exclude_limit,
    limit_filter_pos,
    data_root,
):
    """Cached wrapper for compute_portfolio_returns. intv=-1 means all-intv average."""
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    if intv == -1:
        return compute_portfolio_returns_all_intv(
            by_intv, universe_key, data_root,
            position_method=position_method,
            target_gross=target_gross,
            delay=delay,
            return_source=return_source,
            exclude_limit=exclude_limit,
            limit_filter_pos=limit_filter_pos,
        )
    return compute_portfolio_returns(
        by_intv, universe_key, data_root,
        position_method=position_method,
        target_gross=target_gross,
        intv=intv,
        delay=delay,
        return_source=return_source,
        exclude_limit=exclude_limit,
        limit_filter_pos=limit_filter_pos,
    )


@st.cache_data(show_spinner="计算各 sub-universe PnL 贡献...")
def cached_compute_pnl_by_subuniverse(
    signal_files_by_intv_tuple,
    subuniverse_keys,
    position_method,
    target_gross,
    delay,
    return_source,
    exclude_limit,
    limit_filter_pos,
    data_root,
):
    """Cached wrapper for compute_portfolio_pnl_by_subuniverse."""
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    return compute_portfolio_pnl_by_subuniverse(
        by_intv,
        subuniverse_keys=subuniverse_keys,
        data_root=data_root,
        position_method=position_method,
        target_gross=target_gross,
        delay=delay,
        return_source=return_source,
        exclude_limit=exclude_limit,
        limit_filter_pos=limit_filter_pos,
    )


@st.cache_data(show_spinner="计算各 sub-universe 持仓占比...")
def cached_compute_position_share_by_subuniverse(
    signal_files_by_intv_tuple,
    subuniverse_keys,
    position_method,
    target_gross,
    delay,
    return_source,
    exclude_limit,
    limit_filter_pos,
    data_root,
):
    """Cached wrapper for compute_portfolio_position_share_by_subuniverse."""
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    return compute_portfolio_position_share_by_subuniverse(
        by_intv,
        subuniverse_keys=subuniverse_keys,
        data_root=data_root,
        position_method=position_method,
        target_gross=target_gross,
        delay=delay,
        return_source=return_source,
        exclude_limit=exclude_limit,
        limit_filter_pos=limit_filter_pos,
    )


@st.cache_data(show_spinner="计算每月贡献前 N 的股票...")
def cached_compute_top_stocks_by_month(
    signal_files_by_intv_tuple,
    subuniverse_keys,
    position_method,
    target_gross,
    delay,
    return_source,
    exclude_limit,
    limit_filter_pos,
    top_n,
    data_root,
):
    """Cached wrapper for compute_top_stocks_by_month."""
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    return compute_top_stocks_by_month(
        by_intv,
        subuniverse_keys=subuniverse_keys,
        data_root=data_root,
        position_method=position_method,
        target_gross=target_gross,
        delay=delay,
        return_source=return_source,
        exclude_limit=exclude_limit,
        limit_filter_pos=limit_filter_pos,
        top_n=top_n,
    )


@st.cache_data(show_spinner="计算 ZZ3000 基准收益...")
def cached_compute_benchmark(data_root, intv, delay, return_source, exclude_limit):
    """Cached wrapper for compute_zz3000_benchmark_returns."""
    return compute_zz3000_benchmark_returns(
        data_root, intv=intv, delay=delay, return_source=return_source,
        exclude_limit=exclude_limit,
    )


@st.cache_data(show_spinner="计算信号统计...")
def cached_compute_stats(
    signal_files_by_intv_tuple,
    universe_key,
    intv,
    data_root,
):
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    if intv == -1:
        return compute_signal_statistics_all_intv(
            by_intv, universe_key, data_root
        )
    return compute_signal_statistics(by_intv, universe_key, data_root, intv=intv)


@st.cache_data(show_spinner="计算 daily IC...")
def cached_compute_ic(
    signal_files_by_intv_tuple,
    universe_key,
    intv,
    delay,
    return_source,
    exclude_limit,
    data_root,
):
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    if intv == -1:
        return compute_daily_ic_all_intv(
            by_intv, universe_key, data_root,
            delay=delay, return_source=return_source,
            exclude_limit=exclude_limit,
        )
    return compute_daily_ic(
        by_intv, universe_key, data_root,
        intv=intv, delay=delay, return_source=return_source,
        exclude_limit=exclude_limit,
    )


@st.cache_data(show_spinner="计算 monthly IC...")
def cached_compute_monthly_ic(
    signal_files_by_intv_tuple,
    universe_key,
    intv,
    delay,
    return_source,
    exclude_limit,
    data_root,
):
    by_intv = {k: list(v) for k, v in signal_files_by_intv_tuple}
    if intv == -1:
        return compute_monthly_ic_all_intv(
            by_intv, universe_key, data_root,
            delay=delay, return_source=return_source,
            exclude_limit=exclude_limit,
        )
    return compute_monthly_ic(
        by_intv, universe_key, data_root,
        intv=intv, delay=delay, return_source=return_source,
        exclude_limit=exclude_limit,
    )


# ── date helpers ──────────────────────────────────────────────────────────────

def _to_datetime(date_int):
    """Convert YYYYMMDD int to pandas Timestamp."""
    return pd.to_datetime(str(date_int), format='%Y%m%d')


def _factor_label(signal_root):
    """Return a human-friendly label for a factor: its folder's basename."""
    if not signal_root:
        return "因子"
    name = os.path.basename(os.path.normpath(signal_root))
    return name if name else signal_root


def _render_factor_selector_sidebar(available_factors, factor_labels):
    """Render factor selector in the sidebar at the top.

    The sidebar is always visible on desktop browsers, so the user can
    switch factors from any scroll position.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🔄 当前展示的因子")
        st.radio(
            "选择展示的因子",
            available_factors,
            format_func=lambda x: factor_labels.get(x, x),
            key='active_factor',
        )


# ── UI ────────────────────────────────────────────────────────────────────────

# Universes compared in the multi-universe chart / table
COMPARISON_UNIVERSES = ['whole', 'A500', 'HS300', 'ZZ500', 'ZZ1000', 'ZZ2000', 'SH', 'SZ', 'ChiNext', 'STAR', 'Residual']

UNI_COLORS = {
    'whole': 'royalblue',
    'A500': 'olive',
    'HS300': 'green',
    'ZZ500': 'purple',
    'ZZ1000': 'teal',
    'ZZ2000': 'orange',
    'SH': 'pink',
    'SZ': 'cyan',
    'ChiNext': 'red',
    'STAR': 'brown',
    'Residual': 'gray',
}


def _compute_factor_results(
    signal_root,
    universe_options,
    data_root,
    selected_method,
    target_gross,
    selected_intv,
    selected_delay,
    selected_return_source,
    exclude_limit,
    limit_filter_pos,
):
    """Compute all per-factor results (returns, IC, stats) for one signal root.

    Returns a dict with the computed dataframes / series, or {'error': msg}.
    """
    try:
        by_intv = get_signal_files(signal_root)
    except Exception as e:
        return {'error': f"扫描信号文件失败: {e}"}

    if not by_intv:
        return {'error': f"在 {signal_root} 中未找到信号文件"}

    by_intv_tuple = _to_hashable(by_intv)

    # Per-universe daily returns
    uni_daily_returns = {}
    uni_cum_returns = {}
    uni_monthly_returns = {}

    for uni_key in COMPARISON_UNIVERSES:
        uni_daily = cached_compute_returns(
            by_intv_tuple,
            uni_key,
            selected_method,
            target_gross,
            int(selected_intv),
            selected_delay,
            selected_return_source,
            exclude_limit,
            limit_filter_pos,
            data_root,
        )
        if uni_daily.empty:
            continue
        uni_daily_returns[uni_key] = uni_daily
        uni_cum_returns[uni_key] = compute_cumulative_returns(uni_daily)
        uni_monthly_returns[uni_key] = compute_monthly_returns(uni_daily)

    if not uni_daily_returns:
        return {'error': "未能计算任何 universe 的收益，请检查参数"}

    primary_uni = 'whole' if 'whole' in uni_daily_returns else list(uni_daily_returns.keys())[0]
    daily_returns = uni_daily_returns[primary_uni]
    cum_returns = uni_cum_returns[primary_uni]
    monthly_returns = uni_monthly_returns[primary_uni]

    # ZZ3000 benchmark
    benchmark_daily_full = cached_compute_benchmark(
        data_root, int(selected_intv), selected_delay, selected_return_source,
        exclude_limit,
    )
    if not daily_returns.empty and not benchmark_daily_full.empty:
        factor_min_date = int(daily_returns['date'].min())
        factor_max_date = int(daily_returns['date'].max())
        benchmark_daily = benchmark_daily_full[
            (benchmark_daily_full['date'] >= factor_min_date) &
            (benchmark_daily_full['date'] <= factor_max_date)
        ].reset_index(drop=True)
    else:
        benchmark_daily = benchmark_daily_full
    benchmark_cum = compute_cumulative_returns(benchmark_daily) if not benchmark_daily.empty else pd.Series(dtype=float)

    # Monthly IC per universe
    comparison_monthly_ic = None
    for uni_key in COMPARISON_UNIVERSES:
        uni_monthly_ic = cached_compute_monthly_ic(
            by_intv_tuple,
            uni_key,
            int(selected_intv),
            selected_delay,
            selected_return_source,
            exclude_limit,
            data_root,
        )
        if uni_monthly_ic.empty:
            continue
        uni_monthly_ic = uni_monthly_ic[['month_label', 'monthly_ic']].rename(
            columns={'monthly_ic': universe_options[uni_key]}
        )
        if comparison_monthly_ic is None:
            comparison_monthly_ic = uni_monthly_ic
        else:
            comparison_monthly_ic = comparison_monthly_ic.merge(
                uni_monthly_ic, on='month_label', how='outer'
            )

    # Daily IC per universe
    daily_ic_by_uni = {}
    for uni_key in COMPARISON_UNIVERSES:
        uni_daily_ic = cached_compute_ic(
            by_intv_tuple,
            uni_key,
            int(selected_intv),
            selected_delay,
            selected_return_source,
            exclude_limit,
            data_root,
        )
        if uni_daily_ic.empty:
            continue
        daily_ic_by_uni[uni_key] = uni_daily_ic

    return {
        'daily_returns': daily_returns,
        'cum_returns': cum_returns,
        'monthly_returns': monthly_returns,
        'uni_daily_returns': uni_daily_returns,
        'uni_cum_returns': uni_cum_returns,
        'uni_monthly_returns': uni_monthly_returns,
        'benchmark_daily': benchmark_daily,
        'benchmark_cum': benchmark_cum,
        'comparison_monthly_ic': comparison_monthly_ic,
        'daily_ic_by_uni': daily_ic_by_uni,
        'primary_uni': primary_uni,
        'by_intv': by_intv,
    }


def main():
    st.title("📊 因子信号分析工具")
    st.markdown("---")

    cfg = get_config()
    universe_options = get_universe_options()
    data_root = cfg['data_root']

    # ── sidebar: parameters ───────────────────────────────────────────────────
    st.sidebar.header("⚙️ 参数设置")

    # Factor A (primary) signal root
    signal_root_a = st.sidebar.text_input(
        "因子 A 信号数据目录",
        value=cfg['signal_root'],
        help="存放因子A信号 .npz 文件的根目录",
    )
    factor_a_label = _factor_label(signal_root_a)

    # Optional second factor for comparison
    enable_second_factor = st.sidebar.checkbox(
        "启用因子 B 对比",
        value=False,
        help="勾选后可输入第二个因子信号目录，同时分析两个因子并切换展示。",
    )
    signal_root_b = cfg['signal_root']
    if enable_second_factor:
        signal_root_b = st.sidebar.text_input(
            "因子 B 信号数据目录",
            value=cfg['signal_root'],
            help="存放因子B信号 .npz 文件的根目录",
        )
    factor_b_label = _factor_label(signal_root_b)

    # Use signal_root_a as the primary signal_root for backward-compat
    signal_root = signal_root_a

    try:
        by_intv = get_signal_files(signal_root)
    except Exception as e:
        st.error(f"扫描信号文件失败: {e}")
        return

    if not by_intv:
        st.warning(f"在 {signal_root} 中未找到信号文件")
        return

    available_intvs = sorted(by_intv.keys())
    st.sidebar.markdown(f"**可用 intv**: {available_intvs}")

    # 全intv平均 checkbox
    use_all_intv = st.sidebar.checkbox("全intv平均", value=False,
        help="勾选后，收益、信号统计、Daily IC 将按所有可用 intv 求平均。")

    if use_all_intv:
        selected_intv = -1  # sentinel for all-intv average
        st.sidebar.success("使用全intv平均")
    else:
        # intv selection: number input 0-47
        selected_intv = st.sidebar.number_input(
            "选择 intv (0-47)",
            min_value=0,
            max_value=47,
            value=0,
            step=1,
            help="选择区间索引，0-47。如果该 intv 没有信号文件，将使用小于它的最大的可用 intv 的信号。"
        )
        # Find the best intv (with fallback)
        best_intv = _find_best_intv(available_intvs, int(selected_intv))
        if best_intv is None:
            st.warning(f"intv={selected_intv} 没有可用的信号文件（无小于它的 intv）")
            return
        if best_intv != selected_intv:
            st.sidebar.info(f"intv={selected_intv} 无信号，回退到 intv={best_intv}")
        else:
            st.sidebar.success(f"使用 intv={best_intv}")

    # Signal-trade delay: use tidx+n stock returns to compute returns for tidx signal
    selected_delay = st.sidebar.number_input(
        "信号与交易时间延迟 (0-6)",
        min_value=0,
        max_value=6,
        value=1,
        step=1,
        help="选择信号产生后多少个5分钟间隔后交易。选择 n 表示使用 tidx+n 的股票收益率计算 tidx 时因子信号的收益。"
    )

    # Position method selection (removed top_n; added power)
    position_methods = {
        'signal_weighted': '信号加权 (demean + scale)',
        'power': 'Power (rank → demean → scale)',
    }
    selected_method = st.sidebar.selectbox(
        "持仓方式",
        list(position_methods.keys()),
        index=0,
        format_func=lambda x: position_methods[x],
    )

    target_gross = st.sidebar.number_input(
        "目标总仓位 (gross exposure)",
        min_value=1e4,
        max_value=1e10,
        value=20e6,
        step=1e6,
        format="%.0f",
    )

    # Return source selection: raw (close-based) or dsrt (pre-computed DSRT)
    return_sources = {
        'raw': 'Raw (基于收盘价计算)',
        'dsrt': 'DSRT (使用 CNE5Ret.DSRT 隔日收益)',
    }
    selected_return_source = st.sidebar.selectbox(
        "收益计算方式",
        list(return_sources.keys()),
        index=0,
        format_func=lambda x: return_sources[x],
    )

    # Whether to exclude stocks that hit limit-up/down at the trading tidx
    exclude_limit = st.sidebar.checkbox(
        "剔除涨/跌停股",
        value=False,
        help="勾选后，计算 day=x 的次日收益时会遮盖掉在 day=x+1 当前 tidx 已涨停"
             "或跌停的股票（价格触及当日涨停价/跌停价）。",
    )

    # When exclude_limit is on, choose *when* to apply the limit filter:
    #   - False (default): positions are computed first, then limit-hit stocks'
    #     PnL contributions are zeroed (post-position mask).
    #   - True: limit-hit stocks are excluded *before* position sizing (i.e.
    #     they get zero position). Equivalent to filtering the tradeable
    #     universe first.
    limit_filter_pos = False
    if exclude_limit:
        limit_filter_pos = st.sidebar.checkbox(
            "剔除涨跌停提前到分配仓位前",
            value=False,
            help="勾选后，先剔除 day=x+1 当前 tidx 已涨停/跌停的股票，"
                 "再在剩余股票上分配仓位（pre-position mask）。"
                 "未勾选时，先在全 universe 上分配仓位，再把涨跌停股的 PnL 贡献清零"
                 "（post-position mask）。",
        )

    # Date range display
    min_date, max_date = get_date_range(signal_root)
    if min_date and max_date:
        st.sidebar.markdown(f"**信号日期范围**: {min_date} ~ {max_date}")

    # ── compute button ────────────────────────────────────────────────────────
    compute_clicked = st.sidebar.button("🚀 计算收益", type="primary",
                                         use_container_width=True)

    # Cache key describing all params that affect factor computation, so that
    # changing any of them invalidates the cached factor_results.
    cache_key = (
        signal_root_a, signal_root_b, enable_second_factor,
        selected_method, target_gross, selected_intv,
        selected_delay, selected_return_source, exclude_limit,
        limit_filter_pos,
    )

    # Only compute when the button is explicitly clicked, or when parameters
    # have changed since the last computation. We do NOT auto-compute on first
    # load — the user must press the button.
    prev_key = st.session_state.get('factor_cache_key')
    params_changed = prev_key != cache_key
    has_cached = 'factor_results' in st.session_state

    if compute_clicked or (params_changed and has_cached):
        factors_to_compute = [('A', signal_root_a)]
        if enable_second_factor:
            factors_to_compute.append(('B', signal_root_b))

        factor_results = {}
        for fkey, froot in factors_to_compute:
            label = _factor_label(froot)
            with st.spinner(f"计算因子 {label} ..."):
                factor_results[fkey] = _compute_factor_results(
                    froot, universe_options, data_root,
                    selected_method, target_gross, selected_intv,
                    selected_delay, selected_return_source, exclude_limit,
                    limit_filter_pos,
                )
        st.session_state['factor_results'] = factor_results
        st.session_state['factor_cache_key'] = cache_key
        # Reset the active-factor selection when the set of factors changes.
        st.session_state.pop('active_factor', None)
        factor_results = st.session_state['factor_results']
    elif has_cached:
        factor_results = st.session_state['factor_results']
    else:
        # No computation yet — show a prompt and stop.
        st.info("请在左侧调整参数后点击 🚀 计算收益 按钮。")
        return

    # Surface errors; if no factor computed successfully, stop.
    available_factors = [k for k, v in factor_results.items() if 'error' not in v]
    if not available_factors:
        for fkey, res in factor_results.items():
            label = _factor_label(
                signal_root_a if fkey == 'A' else signal_root_b
            )
            st.error(f"因子 {label}: {res.get('error', '未知错误')}")
        return

    # Toggle: which factor to display. When two factors are available, render
    # the selector in the sidebar (always visible on desktop) and inline at
    # the top of the main content area so the user can switch from any position.
    if len(available_factors) > 1:
        # Build the label mapping: A -> folder name of signal_root_a, etc.
        factor_labels = {
            'A': factor_a_label,
            'B': factor_b_label,
        }
        _render_factor_selector_sidebar(available_factors, factor_labels)
        active_factor = st.session_state.get('active_factor', available_factors[0])
    else:
        active_factor = available_factors[0]

    # Resolve the human-readable label for the active factor.
    if active_factor == 'A':
        active_label = factor_a_label
    else:
        active_label = factor_b_label

    # Show errors for failed factors
    for fkey, res in factor_results.items():
        if 'error' in res:
            label = factor_a_label if fkey == 'A' else factor_b_label
            st.warning(f"因子 {label}: {res['error']}")

    # Pull active factor's results into local vars used by display code below
    active = factor_results[active_factor]
    by_intv_tuple = _to_hashable(active['by_intv'])
    daily_returns = active['daily_returns']
    cum_returns = active['cum_returns']
    monthly_returns = active['monthly_returns']
    uni_daily_returns = active['uni_daily_returns']
    uni_cum_returns = active['uni_cum_returns']
    uni_monthly_returns = active['uni_monthly_returns']
    primary_uni = active['primary_uni']
    benchmark_daily = active['benchmark_daily']
    benchmark_cum = active['benchmark_cum']
    comparison_monthly_ic = active['comparison_monthly_ic']
    daily_ic_by_uni = active['daily_ic_by_uni']

    # ── summary metrics ───────────────────────────────────────────────────────
    st.header("📈 收益分析")

    total_days = len(daily_returns)
    final_cum = cum_returns.iloc[-1] if not cum_returns.empty else 0.0
    avg_daily = daily_returns['portfolio_return'].mean() if not daily_returns.empty else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("总交易日", total_days)
    m2.metric("累计收益", f"{final_cum * 100:.2f}%")
    m3.metric("日均收益", f"{avg_daily * 100:.4f}%")

    # ── cumulative return chart (daily frequency) ─────────────────────────────
    method_label = position_methods[selected_method]
    intv_label = "全intv平均" if use_all_intv else f"intv={selected_intv}"
    chart_title = f"累计收益 — {active_label} | {method_label} | {intv_label}"

    # Available universes with non-empty cumulative returns.
    cum_uni_options = [
        uni_key for uni_key in COMPARISON_UNIVERSES
        if uni_key in uni_cum_returns and not uni_cum_returns[uni_key].empty
    ]
    # Default: all available sub-universes selected.
    cum_selected = st.multiselect(
        "选择要显示的 universe 曲线",
        cum_uni_options,
        default=cum_uni_options,
        format_func=lambda k: universe_options.get(k, k),
        key='cum_uni_selector',
    )

    fig_cum = go.Figure()
    for uni_key in cum_selected:
        if uni_key not in uni_daily_returns:
            continue
        uni_daily = uni_daily_returns[uni_key]
        uni_cum = uni_cum_returns[uni_key]
        if uni_cum.empty:
            continue
        fig_cum.add_trace(go.Scatter(
            x=uni_daily['date'].apply(_to_datetime),
            y=uni_cum * 100,
            mode='lines',
            name=f"{universe_options[uni_key]}",
            line=dict(color=UNI_COLORS.get(uni_key, 'black'), width=2),
            visible=True,
        ))
    # Include benchmark only if at least one universe is shown.
    if cum_selected and not benchmark_cum.empty:
        fig_cum.add_trace(go.Scatter(
            x=benchmark_daily['date'].apply(_to_datetime),
            y=benchmark_cum * 100,
            mode='lines',
            name="ZZ3000 基准",
            line=dict(color='gray', width=1.5, dash='dash'),
        ))

    fig_cum.update_layout(
        title=chart_title,
        xaxis_title="日期",
        yaxis_title="累计收益 (%)",
        hovermode='x unified',
        height=450,
    )
    fig_cum.update_xaxes(
        tickformat='%Y-%m-%d',
        tickangle=-45,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── monthly returns ───────────────────────────────────────────────────────
    st.subheader("📋 月度收益")

    # Universe selector — defaults to the active factor's primary universe.
    bar_uni_options = [
        uni_key for uni_key in COMPARISON_UNIVERSES
        if uni_key in uni_monthly_returns and not uni_monthly_returns[uni_key].empty
    ]
    if bar_uni_options:
        default_idx = (
            bar_uni_options.index(primary_uni)
            if primary_uni in bar_uni_options else 0
        )
        bar_uni_key = st.selectbox(
            "选择 universe 查看月度收益",
            bar_uni_options,
            index=default_idx,
            format_func=lambda k: universe_options.get(k, k),
        )
        bar_monthly = uni_monthly_returns.get(bar_uni_key, monthly_returns)
        if not bar_monthly.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bar_monthly['month_label'],
                y=bar_monthly['monthly_return'] * 100,
                name='月度收益',
                marker_color=['green' if x > 0 else 'red'
                              for x in bar_monthly['monthly_return']],
                text=[f"{x*100:.2f}%" for x in bar_monthly['monthly_return']],
                textposition='outside',
            ))
            fig_bar.update_layout(
                title=f"月度收益柱状图 — {universe_options.get(bar_uni_key, bar_uni_key)}",
                xaxis_title="月份",
                yaxis_title="月度收益 (%)",
                height=400,
            )
            fig_bar.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("无可用 universe 月度收益数据")

    # ── multi-universe monthly comparison table ───────────────────────────────
    st.subheader("📊 多 universe 月度收益对比")
    st.markdown("对比各 universe 在每个月的收益率")

    comparison_monthly = None
    for uni_key in COMPARISON_UNIVERSES:
        if uni_key not in uni_monthly_returns:
            continue
        uni_monthly = uni_monthly_returns[uni_key]
        if uni_monthly.empty:
            continue
        uni_monthly = uni_monthly[['month_label', 'monthly_return']].rename(
            columns={'monthly_return': universe_options[uni_key]}
        )
        if comparison_monthly is None:
            comparison_monthly = uni_monthly
        else:
            comparison_monthly = comparison_monthly.merge(
                uni_monthly, on='month_label', how='outer'
            )

    if comparison_monthly is not None and not comparison_monthly.empty:
        display_comp = comparison_monthly.copy()
        for col in display_comp.columns:
            if col != 'month_label':
                display_comp[col] = display_comp[col] * 100

        styled_comp = display_comp.style.format({
            col: '{:.2f}%' for col in display_comp.columns if col != 'month_label'
        })
        st.dataframe(styled_comp, use_container_width=True)
    else:
        st.warning("无法生成多 universe 月度对比表")

    # ── monthly return share table ───────────────────────────────────────────
    st.subheader("📊 各月收益占比")
    st.markdown(
        "对每个交易日，先按 whole universe 计算仓位，再把当日总 PnL 按各 sub-universe 成员拆解；"
        "按月累计后除以当月因子在 whole universe 上的总收益，得到各 sub-universe 在当月总收益中所占百分比。"
    )

    # Sub-universes that form a partition of the whole universe.
    share_uni_keys = [
        uni_key for uni_key in COMPARISON_UNIVERSES
        if uni_key != 'whole'
    ]

    share_table = None
    if share_uni_keys:
        # Daily PnL contribution per sub-universe, averaged across intvs.
        pnl_by_sub = cached_compute_pnl_by_subuniverse(
            by_intv_tuple,
            tuple(share_uni_keys),
            selected_method,
            target_gross,
            selected_delay,
            selected_return_source,
            exclude_limit,
            limit_filter_pos,
            data_root,
        )

        if not pnl_by_sub.empty:
            # Aggregate daily contributions to monthly per sub-universe.
            pnl_by_sub['month'] = pnl_by_sub['date'].apply(lambda d: month_key(d))
            monthly_pnl = pnl_by_sub.groupby(
                ['month', 'subuniverse']
            ).agg(pnl_contribution=('pnl_contribution', 'sum')).reset_index()

            # Pivot: rows = month_label, cols = sub-universe.
            pivot = monthly_pnl.pivot(
                index='month', columns='subuniverse', values='pnl_contribution'
            )
            # Reorder columns to match share_uni_keys order.
            cols_ordered = [k for k in share_uni_keys if k in pivot.columns]
            pivot = pivot[cols_ordered]
            pivot = pivot.reset_index().rename(columns={'month': 'month_label'})
            pivot['month_label'] = pivot['month_label'].apply(month_label)

            # Denominator: whole-universe monthly return × target_gross, which
            # equals monthly total PnL when sub-universes partition whole.
            whole_monthly = monthly_returns[['month_label', 'monthly_return']].copy()
            whole_monthly['total_pnl'] = whole_monthly['monthly_return'] * target_gross

            merged = pivot.merge(whole_monthly[['month_label', 'total_pnl']],
                                 on='month_label', how='left')

            for col in cols_ordered:
                merged[col] = np.where(
                    merged['total_pnl'].abs() > 1e-12,
                    merged[col] / merged['total_pnl'] * 100,
                    0.0,
                )

            share_table = merged[['month_label'] + cols_ordered]

    if share_table is not None and not share_table.empty:
        styled_share = share_table.style.format({
            col: '{:.2f}%' for col in share_table.columns if col != 'month_label'
        })
        st.dataframe(styled_share, use_container_width=True)
    else:
        st.warning("无法生成各月收益占比表")

    # ── monthly position share table ─────────────────────────────────────────
    st.subheader("📊 每月持仓占比")
    st.markdown(
        "对每个交易日，先按 whole universe 计算仓位，再把当日各 sub-universe 的"
        "持仓绝对值之和除以 target_gross，得到当日持仓占比；按月对每日持仓占比求平均。"
    )

    pos_share_table = None
    if share_uni_keys:
        # Daily position share per sub-universe, averaged across intvs.
        pos_share_by_sub = cached_compute_position_share_by_subuniverse(
            by_intv_tuple,
            tuple(share_uni_keys),
            selected_method,
            target_gross,
            selected_delay,
            selected_return_source,
            exclude_limit,
            limit_filter_pos,
            data_root,
        )

        if not pos_share_by_sub.empty:
            # Average daily position shares within each month.
            pos_share_by_sub['month'] = pos_share_by_sub['date'].apply(
                lambda d: month_key(d)
            )
            monthly_pos_share = pos_share_by_sub.groupby(
                ['month', 'subuniverse']
            ).agg(position_share=('position_share', 'mean')).reset_index()

            pivot = monthly_pos_share.pivot(
                index='month', columns='subuniverse', values='position_share'
            )
            cols_ordered = [k for k in share_uni_keys if k in pivot.columns]
            pivot = pivot[cols_ordered]
            pivot = pivot.reset_index().rename(columns={'month': 'month_label'})
            pivot['month_label'] = pivot['month_label'].apply(month_label)

            for col in cols_ordered:
                pivot[col] = pivot[col] * 100

            pos_share_table = pivot

    if pos_share_table is not None and not pos_share_table.empty:
        styled_pos_share = pos_share_table.style.format({
            col: '{:.2f}%' for col in pos_share_table.columns if col != 'month_label'
        })
        st.dataframe(styled_pos_share, use_container_width=True)
    else:
        st.warning("无法生成每月持仓占比表")

    # ── monthly top-N contributing stocks table ───────────────────────────────
    st.subheader("📊 每月贡献前 5 的股票")
    st.markdown(
        "对每个交易日，先按 whole universe 计算仓位，再计算每只股票的当日 PnL 贡献 "
        "(`position_i * return_i`)。按月累计后，取每月贡献前 5 的股票，"
        "并列出它们所属的 sub-universe。"
    )

    top_stocks_table = cached_compute_top_stocks_by_month(
        by_intv_tuple,
        tuple(share_uni_keys),
        selected_method,
        target_gross,
        selected_delay,
        selected_return_source,
        exclude_limit,
        limit_filter_pos,
        5,
        data_root,
    )

    if top_stocks_table is not None and not top_stocks_table.empty:
        styled_top = top_stocks_table.style.format({
            'pnl_contribution': '{:.2f}',
            'rank': '{:.0f}',
            'position_rank': '{:.0f}',
        })
        st.dataframe(styled_top, use_container_width=True)
    else:
        st.warning("无法生成每月贡献前 5 的股票表")

    # ── monthly IC table ──────────────────────────────────────────────────────
    st.subheader("🎯 各月 IC 对比")
    st.markdown("各月每日因子信号与隔日收益 IC 的月平均值")

    if comparison_monthly_ic is not None and not comparison_monthly_ic.empty:
        styled_ic = comparison_monthly_ic.style.format({
            col: '{:.4f}' for col in comparison_monthly_ic.columns if col != 'month_label'
        })
        st.dataframe(styled_ic, use_container_width=True)
    else:
        st.warning("无法生成月度 IC 对比表")

    # ── daily IC chart with all universes ─────────────────────────────────────
    st.subheader("🎯 Daily IC (rank)")
    st.markdown("各 universe 的每日 rank IC")

    # Available universes for IC chart.
    ic_uni_options = [
        uni_key for uni_key in COMPARISON_UNIVERSES
        if uni_key in daily_ic_by_uni and not daily_ic_by_uni[uni_key].empty
    ]
    ic_selected = st.multiselect(
        "选择要显示的 universe 曲线",
        ic_uni_options,
        default=ic_uni_options,
        format_func=lambda k: universe_options.get(k, k),
        key='ic_uni_selector',
    )

    fig_ic = go.Figure()
    for uni_key in ic_selected:
        if uni_key not in daily_ic_by_uni:
            continue
        uni_daily_ic = daily_ic_by_uni[uni_key]
        fig_ic.add_trace(go.Scatter(
            x=uni_daily_ic['date'].apply(_to_datetime),
            y=uni_daily_ic['ic'],
            mode='lines',
            name=f"{universe_options[uni_key]}",
            line=dict(color=UNI_COLORS.get(uni_key, 'black'), width=1.5),
        ))

    fig_ic.update_layout(
        title=f"Daily Rank IC (各 universe) — {active_label}",
        xaxis_title="日期",
        yaxis_title="IC",
        hovermode='x unified',
        height=450,
    )
    fig_ic.update_xaxes(
        tickformat='%Y-%m-%d',
        tickangle=-45,
    )
    st.plotly_chart(fig_ic, use_container_width=True)

    # ── IC statistics for all universes ───────────────────────────────────────
    st.subheader("📊 IC 统计（各 universe）")
    ic_stats_rows = []
    for uni_key in COMPARISON_UNIVERSES:
        if uni_key not in daily_ic_by_uni:
            continue
        ic_df = daily_ic_by_uni[uni_key]
        if ic_df.empty:
            continue
        mean_ic = ic_df['ic'].mean()
        std_ic = ic_df['ic'].std()
        ir = mean_ic / std_ic if std_ic > 0 else float('nan')
        win_rate = (ic_df['ic'] > 0).mean() * 100
        ic_stats_rows.append({
            'universe': universe_options[uni_key],
            'mean_IC': mean_ic,
            'std_IC': std_ic,
            'IC_IR': ir,
            'IC_win_rate (%)': win_rate,
            'n_days': len(ic_df),
        })

    if ic_stats_rows:
        ic_stats_df = pd.DataFrame(ic_stats_rows)
        styled_ic_stats = ic_stats_df.style.format({
            'mean_IC': '{:.4f}',
            'std_IC': '{:.4f}',
            'IC_IR': '{:.4f}',
            'IC_win_rate (%)': '{:.2f}',
            'n_days': '{:.0f}',
        })
        st.dataframe(styled_ic_stats, use_container_width=True)
    else:
        st.warning("无法计算 IC 统计")

    # ── download data ─────────────────────────────────────────────────────────
    st.header("💾 下载数据")

    # Sanitize label for use in filenames (keep it simple: replace
    # spaces/slashes with underscores so the downloaded file name is safe).
    safe_label = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(active_label)
    ).strip("_") or "factor"

    col1, col2, col3 = st.columns(3)
    with col1:
        if not daily_returns.empty:
            st.download_button(
                "下载日度收益 CSV",
                daily_returns.to_csv(index=False),
                file_name=f"daily_returns_{safe_label}_{selected_method}_intv{selected_intv}.csv",
            )
    with col2:
        if not monthly_returns.empty:
            st.download_button(
                "下载月度收益 CSV",
                monthly_returns.to_csv(index=False),
                file_name=f"monthly_returns_{safe_label}_{selected_method}_intv{selected_intv}.csv",
            )
    with col3:
        if comparison_monthly_ic is not None and not comparison_monthly_ic.empty:
            st.download_button(
                "下载月度 IC CSV",
                comparison_monthly_ic.to_csv(index=False),
                file_name=f"monthly_ic_{safe_label}_intv{selected_intv}.csv",
            )


if __name__ == "__main__":
    main()
