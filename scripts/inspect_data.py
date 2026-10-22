import pandas as pd

news = pd.read_parquet("data/raw/news/AAPL_O.parquet")
px = pd.read_parquet("data/raw/prices/AAPL_O_daily.parquet")

print("NEWS COLUMNS:", list(news.columns))
print(news.head(5).to_string())
print()
print("PRICE COLUMNS:", list(px.columns))
print(px.head(5).to_string())
print()
if "sourceCode" in news.columns:
    print(news["sourceCode"].value_counts().head(10))
print()
print(news.groupby(news["event_time"].dt.date).size().describe())