import typing as t
import argparse
import collections
import os

import tqdm
import pandas as pd
import numpy as np


class UnionFindRank:
    """Union-Find with Path Compression + Union by Rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = n * [0]

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, u, v):
        pu = self.find(u)
        pv = self.find(v)

        if pu == pv:
            return

        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
            return

        if self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
            return

        self.parent[pu] = pv
        self.rank[pv] += 1


def sample_outside_source(subset: pd.Series, complementary_subset: pd.Series, source: str, random_state: t.Optional[int] = None) -> pd.DataFrame:
    n = len(subset)
    m = len(complementary_subset)

    a = subset
    b = complementary_subset.sample(n=n, replace=n > m, random_state=random_state)

    return pd.DataFrame({"sentence_a": a.values, "sentence_b": b.values, "source": n * [source]})


def sample_within_source(subset: pd.DataFrame, source: str, random_state: t.Optional[int] = None) -> pd.DataFrame:
    txt_to_idx = dict()
    subset_lc = subset.applymap(str.lower)

    for _, (a, b, *_) in subset_lc.iterrows():
        if a not in txt_to_idx:
            txt_to_idx[a] = len(txt_to_idx)
        if b not in txt_to_idx:
            txt_to_idx[b] = len(txt_to_idx)

    n = len(txt_to_idx)
    uf = UnionFindRank(n=n)

    for _, (a, b, *_) in subset_lc.iterrows():
        uf.union(txt_to_idx[a], txt_to_idx[b])

    cluster_ids_a = np.array([uf.find(txt_to_idx[a]) for a in subset_lc["sentence_a"].values], dtype=int)
    cluster_ids_b = np.array([uf.find(txt_to_idx[b]) for b in subset_lc["sentence_b"].values], dtype=int)
    cur = np.unique(np.hstack((cluster_ids_a, cluster_ids_b))).size / n

    seeder = cur_random_state = None

    if random_state is not None:
        seeder = np.random.RandomState(random_state)

    all_dfs = []
    pbar = tqdm.tqdm(np.unique(cluster_ids_a), leave=False)

    for cluster_id in pbar:
        connected_component = subset.loc[cluster_ids_a == cluster_id, "sentence_a"]
        disconnected_components = subset.loc[cluster_ids_b != cluster_id, "sentence_b"]
        if seeder is not None:
            cur_random_state = seeder.randint(0, 2**32-1)
        pbar.set_description(f"({source}, {n=}, CUR={cur:.6f}, cur_random_state={cur_random_state})")
        cur_df_negs = sample_outside_source(connected_component, disconnected_components, source=source, random_state=cur_random_state)
        all_dfs.append(cur_df_negs)

    df = pd.concat(all_dfs)
    df.reset_index(inplace=True, drop=True)
    return df


def main(output_name: str, min_within_source_samples: int = 500, debug: bool = False) -> None:
    if not output_name.endswith(".tsv"):
        output_name += ".tsv"

    base_dir = os.path.abspath("./processed_data/ulysses_sbert_1mil_pairs")
    input_uri = os.path.join(base_dir, "positive_pairs.tsv")
    output_uri = os.path.join(base_dir, output_name)

    df = pd.read_csv(input_uri, index_col=0, sep="\t", engine="c", low_memory=False)

    if debug:
        df = df.iloc[::100, :]

    sources = np.unique(df["source"].values)
    negative_pairs: list[pd.DataFrame] = []

    for source in tqdm.tqdm(sources):
        cur_inds = df["source"] == source
        subset = df.loc[cur_inds, ["sentence_a", "sentence_b"]]
        cur_subset_size = np.sum(cur_inds)
        cur_random_state = sum(map(ord, source))
        if cur_subset_size >= min_within_source_samples:
            df_negative = sample_within_source(subset, source, random_state=cur_random_state)
        else:
            df_negative = sample_outside_source(subset["sentence_a"], df.loc[~cur_inds, "sentence_b"], source=source, random_state=cur_random_state)

        negative_pairs.append(df_negative)

        if cur_subset_size > 30000:
            save_dataframe(negative_pairs, output_uri=output_uri)

    save_dataframe(negative_pairs, output_uri=output_uri)


def save_dataframe(negative_pairs: list[pd.DataFrame], output_uri: str):
    df_out = pd.concat(negative_pairs)
    df_out.reset_index(inplace=True, drop=True)
    assert df_out.shape[1] == 3
    df_out.to_csv(output_uri, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample random negatives.")
    parser.add_argument("-o", "--output-name", type=str, default="negative_pairs.tsv")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(output_name=args.output_name, debug=args.debug)
