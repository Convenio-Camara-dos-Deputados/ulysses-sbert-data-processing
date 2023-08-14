import argparse
import os

import pandas as pd


def merge(
    positive_pairs_uri: str,
    negative_pairs_uri: str,
    output_uri: str,
    sep: str,
    sample_size: float | None,
) -> None:
    df_pos = pd.read_csv(os.path.abspath(positive_pairs_uri), sep=sep, index_col=0)
    df_neg = pd.read_csv(os.path.abspath(negative_pairs_uri), sep=sep, index_col=0)

    if sample_size:
        kwargs = {
            "n": int(sample_size) if sample_size >= 1.0 else None,
            "frac": sample_size if sample_size < 1.0 else None,
            "replace": False,
            "axis": "index",
            "ignore_index": True,
        }

        df_pos = df_pos.sample(**kwargs, random_state=1298930)
        df_neg = df_neg.sample(**kwargs, random_state=8)

    df_pos["label"] = 1.0
    df_neg["label"] = 0.0

    df = pd.concat((df_pos, df_neg), ignore_index=True)

    df.to_csv(os.path.abspath(output_uri), sep=sep)
    print(f"Wrote {len(df)} instances to '{output_uri}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("positive_pairs_uri", type=str)
    parser.add_argument("negative_pairs_uri", type=str)
    parser.add_argument("output_uri", type=str)
    parser.add_argument("--sep", default="\t", type=str)
    parser.add_argument("--sample-size", default=None, type=float)
    args = parser.parse_args()
    merge(**vars(args))
