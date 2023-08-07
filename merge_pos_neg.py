import argparse
import os

import pandas as pd


def merge(
    positive_pairs_uri: str, negative_pairs_uri: str, output_uri: str, sep: str = "\t"
) -> None:
    df_pos = pd.read_csv(os.path.abspath(positive_pairs_uri), sep=sep, index_col=0)
    df_neg = pd.read_csv(os.path.abspath(negative_pairs_uri), sep=sep, index_col=0)

    df_pos["label"] = 1.0
    df_neg["label"] = 0.0

    df = pd.concat((df_pos, df_neg), ignore_index=True)

    df.to_csv(os.path.abspath(output_uri), sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("positive_pairs_uri", type=str)
    parser.add_argument("negative_pairs_uri", type=str)
    parser.add_argument("output_uri", type=str)
    parser.add_argument("--sep", "-s", default="\t", type=str)
    args = parser.parse_args()
    merge(**vars(args))
