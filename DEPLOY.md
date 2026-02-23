# 部署指南：www.daniugu.top（阿里云国际 + GitHub）

本文说明如何将强势股筛选器部署到阿里云国际服务器，绑定域名 www.daniugu.top，K 线数据存服务器并每日更新，支持免费试用 1 个月与付费开通。

---

## 一、前置准备

- **阿里云国际**：已购买 ECS（推荐 Ubuntu 22.04）
- **域名**：daniugu.top 已备案或已解析到海外（国际站一般无需备案）
- **GitHub**：代码已推送到 GitHub 仓库

---

## 二、服务器环境

### 2.1 登录 ECS 并安装依赖

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# Python 3.10+
sudo apt install -y python3 python3-pip python3-venv nginx

# 若用 systemd 管理 gunicorn
# 已包含在 Ubuntu 中
```

### 2.2 数据与代码目录建议

- **代码**：`/var/www/stock-app`（或你喜欢的路径）
- **数据**：`/data/gpt`（K 线缓存、stock_list.csv、market_cap.csv 等）
- **用户库**：可放在 `/data/users.db` 或项目下 `data/users.db`

```bash
sudo mkdir -p /data/gpt
sudo chown -R $USER:$USER /data/gpt
mkdir -p /var/www && cd /var/www
```

### 2.3 从 GitHub 拉取代码

```bash
cd /var/www
git clone https://github.com/你的用户名/你的仓库名.git stock-app
cd stock-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

若仓库为私有，请配置 SSH key 或 Personal Access Token 再 clone。

### 2.4 环境变量

在服务器上创建 `/var/www/stock-app/.env`（或 systemd 的 `Environment=`），生产务必设置：

```bash
# 必改：生产密钥
export SECRET_KEY="随机长字符串"

# 数据目录（K 线、股票列表等）
export GPT_DATA_DIR=/data/gpt

# 用户数据库（可选，默认在项目 data/users.db）
export STOCK_APP_DB=/data/users.db

# 管理员账号（充值后在此开通用户），多个用逗号分隔
export ADMIN_USERNAMES=admin
```

首次部署前请把 K 线等数据放到 `/data/gpt`（可从本地上传，或首次在服务器跑一次全量 prefetch）。

---

## 三、Gunicorn + Nginx

### 3.1 Gunicorn 启动

```bash
cd /var/www/stock-app
source venv/bin/activate
gunicorn -w 4 -b 127.0.0.1:8080 "wsgi:app"
```

- `-w 4`：4 个 worker，可按 CPU 调整  
- `-b 127.0.0.1:8080`：只监听本机，由 Nginx 反向代理

### 3.2 Systemd 服务（推荐）

创建 `/etc/systemd/system/stock-app.service`：

```ini
[Unit]
Description=Stock App Gunicorn
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/stock-app
Environment="PATH=/var/www/stock-app/venv/bin"
EnvironmentFile=/var/www/stock-app/.env
ExecStart=/var/www/stock-app/venv/bin/gunicorn -w 4 -b 127.0.0.1:8080 "wsgi:app"
Restart=always

[Install]
WantedBy=multi-user.target
```

若用当前用户运行，把 `User/Group` 改为当前用户。然后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable stock-app
sudo systemctl start stock-app
sudo systemctl status stock-app
```

### 3.3 Nginx 反向代理与域名

创建 `/etc/nginx/sites-available/stock-app`：

```nginx
server {
    listen 80;
    server_name www.daniugu.top daniugu.top;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /var/www/stock-app/static;
    }
}
```

启用并重载：

```bash
sudo ln -s /etc/nginx/sites-available/stock-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 3.4 HTTPS（推荐）

用 certbot 申请 Let’s Encrypt（阿里云国际服务器可直接用）：

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d www.daniugu.top -d daniugu.top
```

按提示选择域名并开启重定向到 HTTPS。证书会自动续期。

---

## 四、域名解析

在域名服务商（如阿里云万网 / 国际）添加 A 记录：

- **主机记录**：`www`（或 `@` 用于根域）
- **记录值**：阿里云 ECS 公网 IP
- **TTL**：600 或默认

确保 `www.daniugu.top` 和（如需）`daniugu.top` 都指向该 IP。

---

## 五、K 线每日更新（仅更新到当日）

K 线缓存在 `GPT_DATA_DIR`（如 `/data/gpt/kline_cache_tencent`）。每天只更新到“当天”的做法：用现有 prefetch 脚本只拉最近约 100 根 K 线，覆盖当日即可。

### 5.1 脚本

项目内已提供：

- `scripts/update_kline_daily.sh`：调用 `scripts.prefetch_kline_tencent`，参数 `--count 100 --max-age-days 0`

### 5.2 Cron

```bash
crontab -e
```

添加（时间可按 A 股收盘后调整，例如 18:05）：

```cron
5 18 * * * export GPT_DATA_DIR=/data/gpt; /var/www/stock-app/scripts/update_kline_daily.sh >> /var/log/stock_kline.log 2>&1
```

或先给脚本执行权限再在 cron 里只写脚本路径，并在脚本内保证 `GPT_DATA_DIR` 已设置。

### 5.3 首次全量

首次部署若 `/data/gpt` 为空，需先跑一次全量（例如 1200 条覆盖 2023 至今），再交给每日任务只更新最近 100 条：

```bash
cd /var/www/stock-app && source venv/bin/activate
export GPT_DATA_DIR=/data/gpt
python -m scripts.prefetch_kline_tencent --count 1200 --max-age-days 0
```

之后每天由 cron 执行 `update_kline_daily.sh` 即可。

---

## 六、用户与计费逻辑

- **免费用户**：注册即享 **1 个月** 试用；试用期内可正常使用筛选与评分。
- **试用结束后**：会跳转到“试用已结束”页，提示充值后联系管理员开通。
- **付费用户**：用户线下/线上充值后，**管理员在后台为该账号点击「开通」**即可。
- **管理员**：通过环境变量 `ADMIN_USERNAMES=admin` 指定（多个用逗号分隔）；管理员可登录后进入「管理后台」，查看用户列表并对已充值用户执行「开通/取消开通」。

流程概括：

1. 用户注册 → 试用 1 个月  
2. 试用结束 → 提示充值并联系管理员  
3. 用户充值 → 管理员在后台为其「开通」  
4. 开通后该用户可继续使用

---

## 七、GitHub 与后续更新

- 代码托管在 GitHub，服务器上 `git pull` 后重启服务即可生效：

```bash
cd /var/www/stock-app
git pull
sudo systemctl restart stock-app
```

- 若要做简单 CI/CD，可在 GitHub Actions 里对 main 分支执行 ssh 到 ECS、git pull、restart service 的步骤（需在 repo 中配置 SSH key 或 token）。

---

## 八、检查清单

- [ ] ECS 安全组放行 80、443
- [ ] 域名 A 记录指向 ECS 公网 IP
- [ ] `SECRET_KEY`、`GPT_DATA_DIR`、`ADMIN_USERNAMES` 已配置
- [ ] `/data/gpt` 下已有 stock_list、K 线缓存等（或已跑过全量 prefetch）
- [ ] Gunicorn（systemd）与 Nginx 正常，HTTPS 已配置
- [ ] 每日 K 线 cron 已添加并测试一次
- [ ] 管理员账号已注册并配置为 `ADMIN_USERNAMES`，能登录并打开管理后台

完成以上步骤后，即可通过 **https://www.daniugu.top** 访问站点，K 线数据存在阿里云服务器并每日更新，免费试用与付费开通流程按上述逻辑运行。
