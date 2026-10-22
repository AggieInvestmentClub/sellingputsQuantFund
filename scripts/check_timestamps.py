import pandas as pd

news = pd.read_parquet("data/raw/news/AAPL_O.parquet")
news["id_date"] = pd.to_datetime(
    news["storyId"].str.extract(r":(\d{8}):")[0],
    format="%Y%m%d", utc=True, errors="coerce")
news["lag_days"] = (news["id_date"] - news["event_time"].dt.normalize()).dt.days
news["is_midnight"] = news["event_time"] == news["event_time"].dt.normalize()

print(news.groupby("sourceCode")["lag_days"].describe()
      .sort_values("count", ascending=False).head(10).to_string())
print()
print(news.groupby("sourceCode")["is_midnight"].mean()
      .sort_values(ascending=False).head(10))
print()
rt = news[news["sourceCode"] == "NS:RTRS"]
print(rt.sample(15, random_state=1)[["event_time", "headline", "storyId"]].to_string())