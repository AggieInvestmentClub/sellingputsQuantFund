import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_news import fetch_headlines, save_news
from src.ingest.lseg_prices import fetch_prices, save_prices

RICS = ["AAPL.O", "MSFT.O", "NVDA.O", "JPM.N", "XOM.N"]
START, END = "2026-09-01", "2026-10-15"

with lseg_session():
    for ric in RICS:
        if Path(f"data/raw/news/{ric.replace('.', '_')}.parquet").exists():
            print(f"{ric}: already downloaded, skipping")
            continue
        try:
            news = fetch_headlines(ric, START, END)
            px = fetch_prices(ric, START, END, "daily")
            print(f"{ric}: {len(news)} headlines, {len(px)} barras diarias")
            if len(news): save_news(ric, news)
            if len(px): save_prices(ric, "daily", px)
        except Exception as e:
            print(f"{ric}: FAILED ({type(e).__name__}): {e}")