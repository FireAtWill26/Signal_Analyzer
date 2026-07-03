# 因子信号分析工具 (Factor Signal Analyzer)

一个用于分析因子信号的工具，支持多种持仓方式、多 universe 对比、收益曲线绘制和信号统计分析。

## 目录结构

```
analyzer/
├── README.md              # 本文档
├── analyzer.md            # 原始任务说明
├── config.json            # 配置文件：路径、universe 列表、价格文件映射
├── data_loader.py         # 数据加载模块：读取 uid/dates/universe mask/价格/信号 npz
├── analyzer.py            # 分析模块：组合收益、月度收益、信号统计、daily IC
├── app.py                 # Streamlit UI 模块（交互式）
├── generate_report.py     # HTML 报告生成器（静态，备选方案）
├── run.sh                 # 启动脚本
├── example_signal/        # 示例信号数据
└── output/                # 输出目录（CSV 下载）
```

## 快速开始

有两种使用方式：

### 方式一：交互式 UI (Streamlit)

```bash
bash /dfs/data/tools/analyzer/run.sh [port]
```

默认端口 8551，启动后浏览器访问 `http://localhost:<port>`。

### 方式二：生成 HTML 报告（无需运行服务器）

```bash
python /dfs/data/tools/analyzer/generate_report.py \
    --signal-dir /dfs/data/tools/analyzer/example_signal \
    --universe ZZ500 \
    --position-method signal_weighted \
    --target-gross 20e6 \
    --adjust forward \
    --intvs i00 \
    --output /dfs/data/tools/analyzer/output/report.html
```

生成后直接用浏览器打开 `report.html` 即可。

### UI 使用说明

1. **左侧边栏设置参数**：
   - 信号数据目录（默认指向 `example_signal/`）
   - 选择 intv 变体（可多选，合并时取均值）
   - 选择 universe（whole/A500/ZZ100/ZZ500/ZZ800/ZZ1000/ZZ1500/ZZ1800/ZZ2000/ZZ3000/HS300）
   - 复权方式（前复权/后复权/不复权）
   - 持仓方式（见下文）
   - 信号日期范围

2. **持仓方式（position_method）**：
   - `信号加权 (signal_weighted)` — **默认**
     - 对信号 demean 使 `sum(positions) ≈ 0`（市场中性）
     - 将 `sum(|positions|)` scale 到 `target_gross`（默认 20e6）
     - 仓位 = 调整后的信号值（货币单位）
     - 日 PnL = `sum(position_i * return_i)`，日收益率 = `PnL / target_gross`
     - 参数：目标总仓位 (gross exposure)
   - `Top-N 等权多头 (top_n)`
     - 在 universe 成员中按信号值选 top-N，等权持有
     - 日收益率 = top-N 股票次日收益的均值
     - 参数：Top-N

3. **主区域查看结果**：
   - 收益分析：累计收益曲线 + 月度收益柱状图 + 月度收益表
   - 多 universe 月度收益对比表
   - **各月收益占比表**（新增）：各 sub-universe 在当月总收益中所占百分比
   - 因子信号统计：mean/std/skew/kurtosis 月度柱状图 + 表
   - Daily IC 曲线 + Mean IC / IC IR / IC Win Rate 指标
   - 下载按钮：日度收益 / 月度收益 / 信号统计 CSV

### 月度收益柱状图（支持 universe 切换）

在 "📋 月度收益" 区段，新增 **"选择 universe 查看月度收益"** 下拉框。用户可切换查看以下任一 universe 的月度收益柱状图：

- whole（全部股票）
- A500（中证A500）
- HS300（沪深300）
- ZZ500（中证500）
- ZZ1000（中证1000）
- ZZ2000（中证2000）
- ChiNext（创业板）
- STAR（科创板）

切换后柱状图标题、数据均会即时更新。

### 各月收益占比表（新增）

**位置**：在 "📊 多 universe 月度收益对比" 表格之后。

**含义**：对每个月，显示各 sub-universe（A500、HS300、ZZ500、ZZ1000、ZZ2000、ChiNext、STAR）在当月因子在 whole universe 上的总收益中所占的百分比。

**计算方式**：

1. **每日 PnL 拆解**：对每个交易日，先按 whole universe 计算仓位（`signal_weighted` 或 `power`），再把当日总 PnL 按各 sub-universe 成员拆解：
   - 对 sub-universe $S$：`pnl_S = sum(positions[mask_S] * returns[mask_S])`
   - 其中 `mask_S` 为 sub-universe $S$ 的成员掩码

2. **按月累计**：将各 sub-universe 的每日 PnL 按月累计，得到 `monthly_pnl_S`

3. **归一化**：除以当月因子在 whole universe 上的总收益：
   - 占比 $= \frac{monthly\_pnl\_S}{monthly\_return\_whole \times target\_gross} \times 100\%$

**注意**：
- 当总收益为 0 或接近 0 时，占比显示为 0%
- 占比可能为负值（当某 sub-universe 月度 PnL 与总收益符号相反时）

## 编程接口

### 数据加载 (`data_loader.py`)

```python
from data_loader import (
    load_config, load_uid, load_dates,
    load_universe_mask, load_close_prices,
    scan_signal_files, load_signal_npz,
    date_to_index, month_key, month_label
)

cfg = load_config()                          # 加载配置
uid = load_uid(cfg['data_root'])             # whole universe 股票代码 (5459,)
dates = load_dates(cfg['data_root'])         # 交易日 (1511,)
mask = load_universe_mask('ZZ500', cfg['data_root'])  # (1511, 5860) 成员掩码
close = load_close_prices(cfg['data_root'], adjust='forward')  # (1511, 5860)
by_intv = scan_signal_files(cfg['signal_root'])  # {intv_tag: [(date, path), ...]}
```

### 分析模块 (`analyzer.py`)

```python
from analyzer import (
    compute_portfolio_returns,       # 统一调度器
    compute_top_n_portfolio_returns, # Top-N 等权多头
    compute_signal_weighted_returns, # 信号加权 (demean + scale)
    compute_cumulative_returns,
    compute_monthly_returns,
    compute_signal_statistics,
    compute_daily_ic,
)

# 统一调度器（默认 signal_weighted）
daily_ret = compute_portfolio_returns(
    signal_files, universe_key, data_root,
    position_method='signal_weighted',  # or 'top_n'
    target_gross=20e6,                  # for signal_weighted
    top_n=50,                           # for top_n
    adjust='forward',
)

# 累计收益
cum = compute_cumulative_returns(daily_ret)

# 月度收益
monthly = compute_monthly_returns(daily_ret)

# 信号统计（mean/std/skew/kurtosis）
stats = compute_signal_statistics(signal_files, universe_key, data_root)

# Daily rank IC
ic = compute_daily_ic(signal_files, universe_key, data_root, adjust='forward')
```

## 数据格式说明

### 信号文件 (npz)

```
example_signal/YYYY/MM/DD/alpha.YYYYMMDD.intv49i<XX>.npz
```

Keys: `alphaV1`, `alphaV2`, `alphaV3`, `vValidTag`, `vStockCode`, `vDumpInfo`

每天有多个 intv 变体（i00/i06/i12/i24/i30/i36），合并时取均值。

### 价格数据 (mmap)

```
ForwardAdjPrices/ForwardAdjPrices.S_DQ_CLOSE.N,5860f           # 不复权
ForwardAdjPrices/ForwardAdjPrices.S_DQ_ADJCLOSE.N,5860f        # 前复权
ForwardAdjPrices/ForwardAdjPrices.S_DQ_ADJCLOSE_BACKWARD.N,5860f  # 后复权
```

- dtype: `<f4` (float32)
- shape: `(1698, 5860)` — 前 1511 行有效（对应 `dates.NI`）

### Universe 成员文件 (mmap)

```
ZZ500/ZZ500.N,5860c   # dtype <i1, shape (1698, 5860), 1=成员
```

注意部分 universe 文件名带 `_` 后缀（如 `ZZ100_.N,5860c`），具体路径见 `config.json`。

### Whole Universe

```
__universe/uid.N128C   # dtype <U32, shape (5459,)
__universe/dates.NI    # dtype <i4, shape (1511,)
```

`uid.N128C` 中的股票代码对应价格/universe 文件的前 5459 列。

## 环境依赖

### 必需包

- `numpy` (>= 1.26)
- `pandas` (>= 2.0)
- `scipy` (>= 1.10)
- `plotly` (>= 6.0) — 用于 UI 绘图
- `streamlit` (>= 1.58) — 用于 UI 框架

### 运行环境

本工具依赖两个 Python 环境：

1. **conda base** (`/root/miniconda3/bin/python`)：包含 `streamlit`、`pandas`、`numpy`
2. **系统 python** (`python3`)：包含 `plotly`、`scipy`、`pandas`、`numpy`

`run.sh` 使用 conda base 的 streamlit 启动 UI。`analyzer.py` 需要 `scipy`（仅系统 python 有），因此在 conda base 中运行时需要额外安装 `scipy` 和 `plotly`：

```bash
/root/miniconda3/bin/conda install -n base -c conda-forge plotly scipy -y
```

## 配置文件 (`config.json`)

```json
{
    "data_root": "/dfs/dataset/230-1724661625521/data/dataprod",
    "signal_root": "/dfs/data/tools/analyzer/example_signal",
    "output_dir": "/dfs/data/tools/analyzer/output",
    "universe_list": [
        {"key": "whole", "name": "Whole Universe", "path": "...", "type": "uid"},
        {"key": "ZZ500", "name": "ZZ500", "path": "...", "type": "membership"}
        // ...
    ],
    "price_files": {
        "raw": "...",
        "forward": "...",
        "backward": "..."
    },
    "intv_variants": ["i00", "i06", "i12", "i24", "i30", "i36"],
    "default_alpha_field": "alphaV1"
}
```

## 持仓方式详解

### 1. 信号加权 (signal_weighted) — 默认

**适用场景**：市场中性策略，信号值直接作为仓位权重。

**计算步骤**：
1. **Demean**：`signal_demeaned = signal - mean(signal)`，使 `sum(positions) ≈ 0`
2. **Scale**：`positions = signal_demeaned * (target_gross / sum(|signal_demeaned|))`，使 `sum(|positions|) = target_gross`
3. **日 PnL**：`pnl = sum(position_i * return_i)`
4. **日收益率**：`portfolio_return = pnl / target_gross`

**参数**：
- `target_gross`：目标总仓位（gross exposure），默认 20e6

**数学验证**：
- `sum(positions) ≈ 0` ✓（多头空头相抵）
- `sum(|positions|) = target_gross` ✓（绝对值总和等于目标）

### 2. Top-N 等权多头 (top_n)

**适用场景**：纯多头策略，选最强信号的 N 只股票等权持有。

**计算步骤**：
1. 在 universe 成员中按信号值选 top-N
2. 次日收益 = `close[t+1] / close[t] - 1`，对 top-N 等权平均
3. 日收益率 = top-N 股票次日收益的均值

**参数**：
- `top_n`：持仓股票数量，默认 50

### 剔除涨/跌停股 (`exclude_limit`)

所有收益计算 / IC 计算函数均支持 `exclude_limit` 参数（默认 `False`）。

- **`exclude_limit=False`**（默认）：保留原有算法，不剔除任何股票。
- **`exclude_limit=True`**：计算 `day=x` 的次日收益时，遮盖掉在 `day=x+1` 当前 tidx 已涨停或跌停的股票（即 `close[x+1, stock, tidx]` 触及当日涨停价 `UpLimPrice` 或跌停价 `DnLimPrice`，容差 `1e-4`）。这些股票的收益率被设为 NaN，进而在 `valid_mask` 中被剔除。

涨跌停价数据来源：`/dfs/dataset/230-1724661625521/data/dataprod/Limits/UpLimPrice.N,5860f` 和 `DnLimPrice.N,5860f`，通过 `data_loader.load_limit_prices(data_root)` 加载，返回 `(up_lim, dn_lim)` 元组。

**注意**：ZZ3000 benchmark 在 `exclude_limit=True` 时会从指数成分中剔除涨停股，由于涨停股权重通常较高，这会显著改变基准收益。如果希望 benchmark 不受此选项影响，可单独调用 `compute_zz3000_benchmark_returns(..., exclude_limit=False)`。

## 示例：命令行使用

```python
import sys
sys.path.insert(0, '/dfs/data/tools/analyzer')

from analyzer import compute_portfolio_returns, compute_cumulative_returns
from data_loader import scan_signal_files, load_config

cfg = load_config()
by_intv = scan_signal_files(cfg['signal_root'])
all_files = by_intv.get('i00', [])

# 默认：信号加权
daily_ret = compute_portfolio_returns(
    all_files, 'ZZ500', cfg['data_root'],
    position_method='signal_weighted',
    target_gross=20e6,
    adjust='forward',
)

cum = compute_cumulative_returns(daily_ret)
print(f"累计收益: {cum.iloc[-1]:.4f}")
```

## generate_report.py 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--signal-dir` | 信号数据根目录（必填） | — |
| `--universe` | Universe 名称 | `ZZ500` |
| `--position-method` | 持仓方式：`signal_weighted` 或 `top_n` | `signal_weighted` |
| `--top-n` | Top-N 股票数（`top_n` 方式用） | `50` |
| `--target-gross` | 目标总仓位（`signal_weighted` 方式用） | `20e6` |
| `--adjust` | 复权方式：`raw`/`forward`/`backward` | `forward` |
| `--intvs` | 逗号分隔的 intv 标签（如 `i00,i36`），留空表示全部 | — |
| `--output` | 输出 HTML 文件路径 | `output/report.html` |

示例：

```bash
# 信号加权（默认），ZZ500 universe
python generate_report.py \
    --signal-dir ./example_signal \
    --universe ZZ500 \
    --position-method signal_weighted \
    --target-gross 20e6

# Top-N 等权多头，HS300 universe，不复权
python generate_report.py \
    --signal-dir ./example_signal \
    --universe HS300 \
    --position-method top_n \
    --top-n 30 \
    --adjust raw

# 多 intv 合并
python generate_report.py \
    --signal-dir ./example_signal \
    --universe ZZ1000 \
    --intvs i00,i06,i12
```

## 故障排查

### 环境依赖说明

本工具依赖两个 Python 环境：

1. **conda base** (`/root/miniconda3/bin/python`)：包含 `streamlit`、`plotly`、`scipy`、`pandas`、`numpy`
2. **系统 python** (`python3`)：包含 `plotly`、`scipy`、`pandas`、`numpy`

`run.sh` 使用 conda base 的 streamlit 启动 UI。`generate_report.py` 可使用任一环境（推荐系统 python）。

### UI 启动报错 `ModuleNotFoundError`

如果报错缺少 `plotly`、`scipy`、`streamlit` 等模块，说明环境不完整。

解决：
```bash
# 安装缺失的包到 conda base
/root/miniconda3/bin/conda install -n base -c conda-forge plotly scipy streamlit -y
```

或者使用 `generate_report.py` 生成 HTML 报告（不依赖 streamlit）：
```bash
python3 /dfs/data/tools/analyzer/generate_report.py --signal-dir ./example_signal --universe ZZ500
```

### 端口被占用

修改 `run.sh` 中的端口号，或：
```bash
bash /dfs/data/tools/analyzer/run.sh 8552
```

### 数据加载失败

检查 `config.json` 中的路径是否正确，特别是 universe 文件名（部分带 `_` 后缀）。

## 开发说明

- `analyzer.py` 中的 `compute_portfolio_returns` 是统一调度器，根据 `position_method` 参数分发到具体实现
- 新增持仓方式：在 `analyzer.py` 中实现 `compute_xxx_returns` 函数，并在调度器中添加分支
- 新增 universe：在 `config.json` 的 `universe_list` 中添加条目
- 缓存：`app.py` 使用 `@st.cache_data` 缓存计算结果，修改参数会自动重新计算
