import glob
import os
import re

import pandas as pd

from . import utils


reg_preproc_comments = re.compile(r"^[^a-zçáéíóúàâôêãẽõü]+|[\"']\s*$", re.IGNORECASE)


def make_pairs(task: str) -> list[tuple[str, str]]:
    uris = glob.glob(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, f"chatgpt/{task}*tsv"))

    assert uris

    fn_preproc_col_b = lambda x: reg_preproc_comments.sub("", x.strip())
    fn_preproc_col_a = fn_preproc_col_b if task == "clusterComments" else None

    dfs = []

    for uri in uris:
        aux = pd.read_csv(uri, sep="\t", index_col=0, header=0, names=["col_a", "col_b"])
        dfs.append(aux)

    df = pd.concat(dfs, ignore_index=True)
    df.dropna(inplace=True)

    if fn_preproc_col_a is not None:
        df.iloc[:, 0] = df.iloc[:, 0].apply(fn_preproc_col_a)

    if fn_preproc_col_b is not None:
        df.iloc[:, 1] = df.iloc[:, 1].apply(fn_preproc_col_b)

    pairs = df.values.tolist()

    assert len(pairs)

    return pairs
