import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

from app.eastmoney import CLIST_URL
from app.paths import GPT_DATA_DIR


def _disable_proxies() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)


def _normalize_code(code: str) -> str:
    value = str(code or "").strip()
    if value.isdigit() and len(value) < 6:
        return value.zfill(6)
    return value


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def fetch_market_caps(
    session: Optional[requests.Session] = None,
    page_size: int = 200,
    max_pages: int = 200,
) -> List[Dict[str, object]]:
    session = session or requests.Session()
    items: List[Dict[str, object]] = []
    seen = set()
    for page in range(1, max_pages + 1):
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f12,f14,f13,f20,f21",
        }
        resp = session.get(CLIST_URL, params=params, timeout=15)
        resp.raise_for_status()
        try:
            payload = resp.json() or {}
        except Exception:
            payload = {}
        data_block = payload.get("data") if isinstance(payload, dict) else {}
        rows = data_block.get("diff", []) if isinstance(data_block, dict) else []
        if not rows:
            break
        added = 0
        for row in rows:
            code = _normalize_code(row.get("f12", ""))
            if not code or code in seen:
                continue
            name = str(row.get("f14", "") or "").strip()
            market = row.get("f13", None)
            total_cap = _safe_float(row.get("f20"))
            float_cap = _safe_float(row.get("f21"))
            items.append(
                {
                    "code": code,
                    "name": name,
                    "market": market if market is not None else "",
                    "total_cap": total_cap,
                    "float_cap": float_cap,
                }
            )
            seen.add(code)
            added += 1
        if added == 0:
            break
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch total market cap snapshot from Eastmoney.")
    parser.add_argument(
        "--out",
        default=os.path.join(GPT_DATA_DIR, "market_cap.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument(
        "--keep-proxy",
        action="store_true",
        help="Keep proxy env vars (default disables proxies).",
    )
    args = parser.parse_args()

    if not args.keep_proxy:
        _disable_proxies()

    items = fetch_market_caps(page_size=args.page_size, max_pages=args.max_pages)
    if not items:
        print("No market cap data fetched.")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["code", "name", "market", "total_cap", "float_cap", "updated_at"],
        )
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["updated_at"] = updated_at
            writer.writerow(row)
    print(f"Saved {len(items)} rows to {args.out}")


if __name__ == "__main__":
    main()
