import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import os
import joblib


def load_build():
    KNN_PATH = os.environ["KNN_PATH"]
    if not os.path.exists(KNN_PATH):
        # appid + ~330 tag-count columns
        tags = pd.read_csv("data/steamspy_tag_data.csv")
        # only what you need
        meta = pd.read_csv("data/steam.csv")[["appid", "name"]]

        # every tag row gains its name
        df = tags.merge(meta, on="appid", how="left")
        df = df.dropna(subset=["name"]).reset_index(
            drop=True)  # drop tag rows with no name match

        # split: coordinates vs coat-check tickets
        tag_cols = [c for c in df.columns if c not in ("appid", "name")]
        # the count matrix -> goes to TF-IDF
        X = df[tag_cols]
        # row-aligned to X, for result lookup
        labels = df[["appid", "name"]]

        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X)

        nn = NearestNeighbors(metric='cosine')
        nn.fit(X)
        joblib.dump(
            {"nn": nn, "tfidf": tfidf, "labels": labels, "tag_cols": tag_cols}, KNN_PATH)

    bundle: dict = joblib.load(KNN_PATH)
    return bundle
