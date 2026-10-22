import pandas as pd

news = pd.read_parquet("data/raw/news/AAPL_O.parquet")
rt = news[news["sourceCode"] == "NS:RTRS"].copy()
rt["base_id"] = rt["storyId"].str.rsplit(":", n=1).str[0]
rt["version"] = rt["storyId"].str.rsplit(":", n=1).str[1]

counts = rt.groupby("base_id").size()
print("Reuters rows:", len(rt), "| unique base IDs:", len(counts))
print("Base IDs with >1 version:", (counts > 1).sum())
print()
multi = counts[counts > 1].index[:3]
print(rt[rt["base_id"].isin(multi)]
      .sort_values(["base_id", "event_time"])
      [["base_id", "version", "event_time", "headline"]].to_string())