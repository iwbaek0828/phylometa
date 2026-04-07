from __future__ import annotations

import pandas as pd


def check_tip_metadata_overlap(tip_names: list[str], meta: pd.DataFrame, id_column: str) -> dict:
    if id_column not in meta.columns:
        raise ValueError(f"ID column '{id_column}' not found in metadata.")

    if meta[id_column].duplicated().any():
        dup = meta.loc[meta[id_column].duplicated(), id_column].astype(str).tolist()
        raise ValueError(f"Duplicate IDs found in metadata: {dup[:10]}")

    meta_ids = set(meta[id_column].astype(str))
    tip_ids = set(map(str, tip_names))

    shared = sorted(tip_ids & meta_ids)
    only_tree = sorted(tip_ids - meta_ids)
    only_meta = sorted(meta_ids - tip_ids)

    return {
        "n_tree_tips": len(tip_ids),
        "n_metadata_ids": len(meta_ids),
        "n_shared": len(shared),
        "only_tree": only_tree,
        "only_metadata": only_meta,
    }
