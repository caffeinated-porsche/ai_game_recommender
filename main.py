import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors

tags = pd.read_csv("data/steamspy_tag_data.csv")        # appid + ~330 tag-count columns
meta = pd.read_csv("data/steam.csv")[["appid", "name"]] # only what you need

df = tags.merge(meta, on="appid", how="left")      # every tag row gains its name
df = df.dropna(subset=["name"]).reset_index(drop=True)  # drop tag rows with no name match

# split: coordinates vs coat-check tickets
tag_cols = [c for c in df.columns if c not in ("appid", "name")]
X = df[tag_cols]                                   # the count matrix -> goes to TF-IDF
labels = df[["appid", "name"]]                      # row-aligned to X, for result lookup


tfidf = TfidfTransformer()
X = tfidf.fit_transform(X)

nn = NearestNeighbors(metric='cosine')
nn.fit(X)