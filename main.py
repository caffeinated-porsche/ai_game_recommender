import pandas as pd
from load_build import load_build
from query import get_tags


def main():
    bundle = load_build()
    tag_cols = bundle["tag_cols"]

    while True:
        notes = input(
            "\nDescribe the type of game you'd like (or 'q' to quit): ").strip()
        if notes == "q":
            break

        tags = get_tags(notes, tag_cols)

        query = pd.DataFrame([[0]*len(tag_cols)], columns=tag_cols)
        query[tags["want"]] = 1
        q = bundle["tfidf"].transform(query)

        _, indices = bundle["nn"].kneighbors(q, n_neighbors=50)   # over-fetch

        avoid = tags["avoid"]
        counts = bundle["counts"]
        results = []
        for idx in indices[0]:
            if avoid and counts.iloc[idx][avoid].sum() > 0:
                continue
            results.append(bundle["labels"].iloc[idx]["name"])
            if len(results) == 10:
                break

        print("Here are the top 10 recommendations we think you'll like:")
        for name in results:
            print(name)


if __name__ == "__main__":
    main()
