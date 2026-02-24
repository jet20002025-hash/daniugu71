# K 线数据源说明

本项目的「今日 / 日 K」数据可从多处获取，当某接口失败时可换源或使用批量更新脚本轮流尝试。

## 脚本内置数据源（`scripts/update_kline_cache.py`）

| 来源     | `--source`   | 说明 |
|----------|--------------|------|
| 新浪     | `sina`       | 新浪财经 K 线接口，易限流，建议 `--delay 0.15～0.3` |
| 网易     | `netease`    | 网易财经 chddata，GBK 编码 |
| 东财     | `eastmoney` | 东方财富 push2his，需 `secid`（如 0.600519） |
| 腾讯     | `tencent`   | 腾讯 ifzq K 线，部分环境可能只到 T-1 |
| AKShare  | `akshare`   | 需 `pip install akshare`，聚合多源，适合作兜底 |

- **默认（`--source auto`）**：按顺序尝试 新浪 → 网易 → 东财 → 腾讯 →（若已装）AKShare，任一成功即写入统一 `code.csv` 缓存。
- 已有完整历史时，仅拉最近约 10 根并合并，不重拉全量。

## 其他可考虑的来源（需自行对接）

- **Tushare**：需注册拿 token，积分制，数据全、稳定。  
  https://tushare.pro/
- **麦瑞 API**：免费申请证书，日 K / 分时等。  
  https://www.mairui.club/gratis.html
- **同花顺 iFinD**：商业数据，需授权。
- **聚宽 / 米筐**：量化平台，有免费额度，接口与策略绑定。

若所有内置源都失败，可先试 `--source akshare`（并安装 akshare），或检查网络/代理/防火墙后再重试各源。
