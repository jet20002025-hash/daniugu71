# 强势股右侧筛选器（本地网页）

本项目会自动拉取 A 股日线数据，按照右侧强势规则筛选候选列表，并在本地网页展示。

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器打开：`http://127.0.0.1:5000`

## 数据与回测规范（必读）

**禁止使用未来数据。** 所有训练、选股、回测、测试只能使用当日及历史数据；用未来数据得到的回测结果无参考价值。详见 [DATA_RULES.md](DATA_RULES.md)。

## 说明

- **主规则模型**：当前默认使用 **mode9**（由 71 倍/mode3 升级，含均线整齐度、量比按比例扣分等），详见 [docs/mode9模型说明.md](docs/mode9模型说明.md)；71 倍原版保留为 mode3 可选。
- 首次运行会缓存 K 线到 `data/kline_cache/`，后续会增量更新。
- 结果会写入 `data/results/latest.json` 与 `data/results/latest.csv`。
- 仅做强势筛选，不提供买点或荐股建议。

## ML 版本（量化买点）

项目新增了一个 ML 版筛选流程，规则版保留不变。

### 训练模型

```bash
python scripts/train_ml.py
```

训练完成后会生成：

- `data/models/ml_model.pkl`
- `data/models/ml_model_meta.json`

### 使用 ML 筛选

在网页表单中选择 “ML 版（量化买点）”，或执行：

```bash
python scripts/scan_ml.py
```

可指定截止日期（历史回看）：

```bash
python scripts/scan_ml.py --end-date 2026-01-31
```

也可指定开始/截止日期范围（按信号日过滤）：

```bash
python scripts/scan_ml.py --start-date 2026-01-01 --end-date 2026-01-31
```
### 使用本地缓存

页面中可选择 “本地缓存（不联网）”，此模式只读取 `data/kline_cache/` 中已有股票。

也可选择 “gpt股票本地库”，读取 `data/gpt/kline_cache_tencent` 的缓存数据（默认路径，可通过环境变量 `GPT_DATA_DIR` 覆盖）。

### 在线更新数据源

在线更新支持多个来源：东财、腾讯、新浪（网易待接入）。

可勾选 “优先本地” 来避免重复联网（已有缓存则直接使用）。

ML 版支持信号强度切换：激进 / 宽松（建议切换后重新训练模型）。

## 大牛股训练（1个月翻倍）

定义：买入点后 20 个交易日内最高价达到买入价 2 倍及以上。

### 查找大牛股信号

```bash
python scripts/find_bull_stocks.py
```

### 训练大牛股模型

```bash
python scripts/train_bull_ml.py --signal-type relaxed
```

### 全量预拉 K 线（写入本地缓存）

```bash
python scripts/prefetch_kline.py
```

可选参数示例：

```bash
python scripts/prefetch_kline.py --count 200 --workers 8 --page-size 100 --max-pages 200
```

可指定在线数据源：

```bash
python scripts/prefetch_kline.py --provider tencent
```

### ML 版本默认假设

- 买点：量化信号当日 `T`，买入价为 `T+1` 开盘价（激进版本）
- 持有周期：40 个交易日
- 标签阈值：收益 ≥ 8% 且跑赢指数 ≥ 3%
- 训练范围：自 2023-01-01 起的日线数据（实际受单次拉取天数限制）
