#!/usr/bin/env bash
# Re-download the raw inputs. Run from repo root: bash data/fetch.sh
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL -o SOXL_OHLC.csv https://raw.githubusercontent.com/jabcho10/dailyorder/main/data/SOXL_OHLC.csv
curl -sSL -o phase5_close_qqq.parquet https://raw.githubusercontent.com/phuazz/breadth-thrust-etf/main/data/phase5_close_qqq.parquet
curl -sSL -o holdings_prices_1y.json https://raw.githubusercontent.com/phuazz/breadth-thrust-etf/main/data/holdings_prices_1y.json
tail -1 SOXL_OHLC.csv
