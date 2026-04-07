from __future__ import annotations

import pandas as pd
from Bio import Phylo


def read_tree(tree_file: str):
    return Phylo.read(tree_file, "newick")


def read_metadata(meta_file: str, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        sep = "," if meta_file.endswith(".csv") else "\t"
    return pd.read_csv(meta_file, sep=sep)


def get_tip_names(tree) -> list[str]:
    return [term.name for term in tree.get_terminals() if term.name is not None]
