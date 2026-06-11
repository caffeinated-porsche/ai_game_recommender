import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import os
from dotenv import load_dotenv
import joblib

load_dotenv()


def load_build():
    KNN_PATH = os.environ["KNN_PATH"]
    if not os.path.exists(KNN_PATH):
        HERE = os.path.dirname(os.path.abspath(__file__))
        tags = pd.read_csv(os.path.join(HERE, "data", "steamspy_tag_data.csv"))
        meta = pd.read_csv(os.path.join(
            HERE, "data", "steam.csv"))[["appid", "name"]]

        df = tags.merge(meta, on="appid", how="left")
        df = df.dropna(subset=["name"]).reset_index(
            drop=True)

        tag_cols = [c for c in df.columns if c not in ("appid", "name")]
        X = df[tag_cols]
        labels = df[["appid", "name"]]

        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X)

        nn = NearestNeighbors(metric='cosine')
        nn.fit(X)
        joblib.dump(
            {"nn": nn, "tfidf": tfidf, "labels": labels,
             "tag_cols": tag_cols, "counts": df[tag_cols].reset_index(drop=True)},
            KNN_PATH)

    bundle: dict = joblib.load(KNN_PATH)
    return bundle
