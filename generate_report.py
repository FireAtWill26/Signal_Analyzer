"""generate_report.py — generate a standalone HTML report for factor signal analysis.

This is an alternative to the Streamlit UI (app.py). It generates a single
HTML file with embedded Plotly charts that can be opened directly in a browser.

Usage:
  python /dfs/data/tools/analyzer/generate_report.py \\
      --signal-dir /dfs/data/tools/analyzer/example_signal \\
      --universe ZZ500 \\
      --position-method signal_weighted \\
      --target-gross 20e6 \\
      --adjust forward \\
      --output /dfs/data/tools/analyzer/output/report.html
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the analyzer directory to path
ANALYZER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ANALYZER_DIR)

from data_loader import (
    load_config, scan_signal_files, load_uid, load_dates,
    load_universe_mask, load_close_prices, load_signal_npz,
    date_to_index, month_key, month_label
)
from analyzer import (
    compute_portfolio_returns, compute_cumulative_returns,
    compute_monthly_returns, compute_signal_statistics, compute_daily_ic
)


def generate_report(
    signal_dir,
    universe_key,
    position_method='signal_weighted',
    top_n=50,
    target_gross=20e6,
    adjust='forward',
    intv_filter=None,
    output_path=None,
):
    """Generate HTML report with factor signal analysis.

    Args:
        signal_dir: root directory containing signal .npz files.
        universe_key: universe to analyze (e.g. 'ZZ500').
        position_method: 'signal_weighted' or 'top_n'.
        top_n: number of top stocks (for 'top_n' method).
        target_gross: target gross exposure (for 'signal_weighted' method).
        adjust: price adjustment type ('raw', 'forward', 'backward').
        intv_filter: list of intv tags to include.
        output_path: path to save HTML report. If None, uses default.

    Returns:
        Path to the generated HTML report.
    """
    cfg = load_config()
    data_root = cfg['data_root']

    # Scan signal files
    by_intv = scan_signal_files(signal_dir)
    if not by_intv:
        raise ValueError(f"No signal files found in {signal_dir}")

    # Collect signal files based on intv filter
    if intv_filter is None:
        all_files = []
        for tag in by_intv.keys():
            all_files.extend(by_intv[tag])
    else:
        all_files = []
        for tag in intv_filter:
            all_files.extend(by_intv.get(tag, []))

    if not all_files:
        raise ValueError("No signal files found for the selected intv variants")

    # Compute portfolio returns
    daily_returns = compute_portfolio_returns(
        all_files, universe_key, data_root,
        position_method=position_method,
        top_n=top_n,
        target_gross=target_gross,
        adjust=adjust,
    )

    # Compute cumulative returns
    cum_returns = compute_cumulative_returns(daily_returns)

    # Compute monthly returns
    monthly_returns = compute_monthly_returns(daily_returns)

    # Compute signal statistics
    signal_stats = compute_signal_statistics(all_files, universe_key, data_root)

    # Compute daily IC
    daily_ic = compute_daily_ic(all_files, universe_key, data_root, adjust=adjust)

    # Generate HTML report
    html = _generate_html(
        daily_returns, cum_returns, monthly_returns,
        signal_stats, daily_ic,
        universe_key, position_method, top_n, target_gross, adjust,
    )

    # Save report
    if output_path is None:
        output_path = os.path.join(cfg['output_dir'], 'report.html')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to: {output_path}")
    return output_path


def _generate_html(
    daily_returns, cum_returns, monthly_returns,
    signal_stats, daily_ic,
    universe_key, position_method, top_n, target_gross, adjust,
):
    """Generate the HTML report content."""
    # Portfolio return chart
    fig1 = go.Figure()
    if not cum_returns.empty:
        fig1.add_trace(go.Scatter(
            x=daily_returns['date'],
            y=cum_returns,
            mode='lines',
            name='累计收益',
            line=dict(color='blue', width=2),
        ))
    fig1.update_layout(
        title=f"累计收益 ({universe_key}, {position_method})",
        xaxis_title="日期",
        yaxis_title="累计收益",
        height=400,
    )

    # Monthly returns bar chart
    fig2 = go.Figure()
    if not monthly_returns.empty:
        colors = ['green' if x > 0 else 'red' for x in monthly_returns['monthly_return']]
        fig2.add_trace(go.Bar(
            x=monthly_returns['month_label'],
            y=monthly_returns['monthly_return'],
            name='月度收益',
            marker_color=colors,
        ))
    fig2.update_layout(
        title="月度收益",
        xaxis_title="月份",
        yaxis_title="月度收益",
        height=350,
    )

    # Signal statistics subplots
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean', 'Std', 'Skew', 'Kurtosis'),
    )
    if not signal_stats.empty:
        fig3.add_trace(go.Bar(x=signal_stats['month_label'], y=signal_stats['mean'], name='Mean'), row=1, col=1)
        fig3.add_trace(go.Bar(x=signal_stats['month_label'], y=signal_stats['std'], name='Std'), row=1, col=2)
        fig3.add_trace(go.Bar(x=signal_stats['month_label'], y=signal_stats['skew'], name='Skew'), row=2, col=1)
        fig3.add_trace(go.Bar(x=signal_stats['month_label'], y=signal_stats['kurtosis'], name='Kurtosis'), row=2, col=2)
    fig3.update_layout(height=500, showlegend=False, title="月度信号统计")

    # Daily IC chart
    fig4 = go.Figure()
    if not daily_ic.empty:
        fig4.add_trace(go.Scatter(
            x=daily_ic['date'],
            y=daily_ic['ic'],
            mode='lines',
            name='Daily IC',
            line=dict(color='orange', width=1),
        ))
        cum_ic = daily_ic['ic'].expanding().mean()
        fig4.add_trace(go.Scatter(
            x=daily_ic['date'],
            y=cum_ic,
            mode='lines',
            name='Cumulative Mean IC',
            line=dict(color='blue', width=2),
        ))
    fig4.update_layout(
        title="Daily Rank IC",
        xaxis_title="日期",
        yaxis_title="IC",
        height=350,
    )

    # Generate tables HTML
    daily_table = daily_returns.to_html(index=False, float_format='%.6f', border=0, classes='table table-striped')
    monthly_table = monthly_returns.to_html(index=False, float_format='%.6f', border=0, classes='table table-striped')
    stats_table = signal_stats.to_html(index=False, float_format='%.6f', border=0, classes='table table-striped')
    ic_table = daily_ic.to_html(index=False, float_format='%.6f', border=0, classes='table table-striped')

    # Summary metrics
    total_days = len(daily_returns)
    final_cum_return = f"{cum_returns.iloc[-1]:.4f}" if not cum_returns.empty else "N/A"
    mean_ic = f"{daily_ic['ic'].mean():.4f}" if not daily_ic.empty and 'ic' in daily_ic.columns else "N/A"
    ic_ir = f"{daily_ic['ic'].mean() / daily_ic['ic'].std():.4f}" if not daily_ic.empty and daily_ic['ic'].std() > 0 else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>因子信号分析报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
        }}
        .metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #ecf0f1;
            padding: 15px 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
            min-width: 150px;
        }}
        .metric-card .label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .params {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            font-size: 14px;
        }}
        .params strong {{
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart {{
            margin: 20px 0;
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>📊 因子信号分析报告</h1>

        <div class="params">
            <strong>参数:</strong><br>
            Universe: {universe_key} |
            持仓方式: {position_method} |
            复权方式: {adjust} |
            {'Top-N: ' + str(top_n) if position_method == 'top_n' else '目标总仓位: ' + str(target_gross)}
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="label">总交易日</div>
                <div class="value">{total_days}</div>
            </div>
            <div class="metric-card">
                <div class="label">累计收益</div>
                <div class="value">{final_cum_return}</div>
            </div>
            <div class="metric-card">
                <div class="label">Mean IC</div>
                <div class="value">{mean_ic}</div>
            </div>
            <div class="metric-card">
                <div class="label">IC IR</div>
                <div class="value">{ic_ir}</div>
            </div>
        </div>

        <h2>📈 累计收益曲线</h2>
        <div class="chart" id="chart1"></div>

        <h2>📊 月度收益</h2>
        <div class="chart" id="chart2"></div>
        {monthly_table}

        <h2>📋 日度收益明细</h2>
        {daily_table}

        <h2>📊 因子信号统计</h2>
        <div class="chart" id="chart3"></div>
        {stats_table}

        <h2>🎯 Daily IC</h2>
        <div class="chart" id="chart4"></div>
        {ic_table}
    </div>

    <script>
        var fig1 = {fig1.to_json()};
        var fig2 = {fig2.to_json()};
        var fig3 = {fig3.to_json()};
        var fig4 = {fig4.to_json()};

        Plotly.newPlot('chart1', fig1.data, fig1.layout, {{responsive: true}});
        Plotly.newPlot('chart2', fig2.data, fig2.layout, {{responsive: true}});
        Plotly.newPlot('chart3', fig3.data, fig3.layout, {{responsive: true}});
        Plotly.newPlot('chart4', fig4.data, fig4.layout, {{responsive: true}});
    </script>
</body>
</html>"""
    return html


def main():
    ap = argparse.ArgumentParser(description='Generate factor signal analysis HTML report')
    ap.add_argument('--signal-dir', required=True,
                    help='Root directory containing signal .npz files')
    ap.add_argument('--universe', default='ZZ500',
                    help='Universe key (e.g. ZZ500, HS300, whole)')
    ap.add_argument('--position-method', default='signal_weighted',
                    choices=['signal_weighted', 'top_n'],
                    help='Position sizing method')
    ap.add_argument('--top-n', type=int, default=50,
                    help='Number of top stocks (for top_n method)')
    ap.add_argument('--target-gross', type=float, default=20e6,
                    help='Target gross exposure (for signal_weighted method)')
    ap.add_argument('--adjust', default='forward',
                    choices=['raw', 'forward', 'backward'],
                    help='Price adjustment type')
    ap.add_argument('--intvs', default='',
                    help='Comma-separated intv tags to include (e.g. i00,i36)')
    ap.add_argument('--output', default=None,
                    help='Output HTML file path')
    args = ap.parse_args()

    intv_filter = [t.strip() for t in args.intvs.split(',')] if args.intvs else None

    generate_report(
        signal_dir=args.signal_dir,
        universe_key=args.universe,
        position_method=args.position_method,
        top_n=args.top_n,
        target_gross=args.target_gross,
        adjust=args.adjust,
        intv_filter=intv_filter,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
