from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


@dataclass
class CladeRecord:
    clade_id: str
    n_tips: int
    n_trait_positive: int
    n_background_positive: int
    odds_ratio: float
    p_value: float


def benjamini_hochberg(pvalues: List[float]) -> List[float]:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out.tolist()


def _prepare_trait_table(
    dist_df: pd.DataFrame,
    meta: pd.DataFrame,
    id_column: str,
    trait: str,
) -> pd.Series:
    if trait not in meta.columns:
        raise ValueError(f"Trait column '{trait}' not found.")
    df = meta[[id_column, trait]].dropna().copy()
    df[id_column] = df[id_column].astype(str)
    df = df[df[id_column].isin(dist_df.index)].copy()
    if df.empty:
        raise ValueError("No overlapping samples between metadata and distance matrix.")
    labels = df.set_index(id_column)[trait].astype(str)
    if labels.nunique() < 2:
        raise ValueError("Categorical trait must contain at least two groups.")
    return labels


def mean_within_group_distance(dist_df: pd.DataFrame, labels: pd.Series) -> float:
    labels = labels.astype(str)
    values = []
    for group in labels.unique():
        members = labels.index[labels == group].tolist()
        if len(members) < 2:
            continue
        sub = dist_df.loc[members, members].values
        iu = np.triu_indices_from(sub, k=1)
        values.extend(sub[iu].tolist())
    if not values:
        return np.nan
    return float(np.mean(values))


def between_group_distance(dist_df: pd.DataFrame, labels: pd.Series) -> float:
    labels = labels.astype(str)
    groups = labels.unique().tolist()
    values = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = labels.index[labels == groups[i]].tolist()
            g2 = labels.index[labels == groups[j]].tolist()
            sub = dist_df.loc[g1, g2].values
            values.extend(sub.ravel().tolist())
    if not values:
        return np.nan
    return float(np.mean(values))


def categorical_clustering_test(
    dist_df: pd.DataFrame,
    meta: pd.DataFrame,
    id_column: str,
    trait: str,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    labels = _prepare_trait_table(dist_df, meta, id_column, trait)
    subdist = dist_df.loc[labels.index, labels.index]

    observed_within = mean_within_group_distance(subdist, labels)
    observed_between = between_group_distance(subdist, labels)
    observed_delta = observed_between - observed_within

    rng = np.random.default_rng(seed)
    perm_within = []
    perm_delta = []
    arr = labels.to_numpy().copy()

    for _ in range(n_perm):
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        shuffled_labels = pd.Series(shuffled, index=labels.index)
        w = mean_within_group_distance(subdist, shuffled_labels)
        b = between_group_distance(subdist, shuffled_labels)
        perm_within.append(w)
        perm_delta.append(b - w)

    perm_within = np.array(perm_within, dtype=float)
    perm_delta = np.array(perm_delta, dtype=float)

    p_clustering = (np.sum(perm_within <= observed_within) + 1) / (len(perm_within) + 1)
    p_separation = (np.sum(perm_delta >= observed_delta) + 1) / (len(perm_delta) + 1)

    return {
        "trait": trait,
        "n_samples": int(len(labels)),
        "n_groups": int(labels.nunique()),
        "observed_mean_within_distance": float(observed_within),
        "observed_mean_between_distance": float(observed_between),
        "observed_delta_between_minus_within": float(observed_delta),
        "permuted_mean_within": float(np.nanmean(perm_within)),
        "permuted_mean_delta": float(np.nanmean(perm_delta)),
        "p_value_clustering": float(p_clustering),
        "p_value_group_separation": float(p_separation),
        "interpretation": "Smaller within-group distance and larger between-minus-within delta indicate stronger phylogenetic structure.",
    }


def _terminal_descendants(clade) -> List[str]:
    return [t.name for t in clade.get_terminals() if t.name is not None]


def clade_enrichment_test(
    tree,
    meta: pd.DataFrame,
    id_column: str,
    trait: str,
    target_value: str,
    min_clade_size: int = 3,
) -> pd.DataFrame:
    df = meta[[id_column, trait]].dropna().copy()
    df[id_column] = df[id_column].astype(str)
    df[trait] = df[trait].astype(str)
    df = df.set_index(id_column)

    universe = set(df.index)
    positives = set(df.index[df[trait] == str(target_value)])
    negatives = universe - positives

    if len(positives) == 0:
        raise ValueError(f"No samples found for trait value '{target_value}'.")

    records = []
    clade_idx = 0

    for clade in tree.find_clades(order="level"):
        members = [x for x in _terminal_descendants(clade) if x in universe]
        if len(members) < min_clade_size or len(members) == len(universe):
            continue

        member_set = set(members)
        a = len(member_set & positives)
        b = len(member_set & negatives)
        c = len(positives - member_set)
        d = len(negatives - member_set)

        if a == 0:
            continue

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        clade_idx += 1
        records.append(
            CladeRecord(
                clade_id=f"clade_{clade_idx}",
                n_tips=len(member_set),
                n_trait_positive=a,
                n_background_positive=len(positives),
                odds_ratio=float(odds_ratio),
                p_value=float(p_value),
            )
        )

    out = pd.DataFrame([r.__dict__ for r in records])
    if out.empty:
        return pd.DataFrame(
            columns=["clade_id", "n_tips", "n_trait_positive", "n_background_positive", "odds_ratio", "p_value", "q_value"]
        )

    out = out.sort_values(["p_value", "odds_ratio"], ascending=[True, False]).reset_index(drop=True)
    out["q_value"] = benjamini_hochberg(out["p_value"].tolist())
    return out


def batch_categorical_tests(
    dist_df: pd.DataFrame,
    meta: pd.DataFrame,
    id_column: str,
    traits: List[str],
    n_perm: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    results = []
    for trait in traits:
        try:
            results.append(categorical_clustering_test(dist_df, meta, id_column, trait, n_perm=n_perm, seed=seed))
        except Exception as exc:
            results.append({"trait": trait, "error": str(exc)})

    out = pd.DataFrame(results)
    if "p_value_clustering" in out.columns:
        mask = out["p_value_clustering"].notna()
        if mask.any():
            out.loc[mask, "q_value_clustering"] = benjamini_hochberg(out.loc[mask, "p_value_clustering"].tolist())
    if "p_value_group_separation" in out.columns:
        mask = out["p_value_group_separation"].notna()
        if mask.any():
            out.loc[mask, "q_value_group_separation"] = benjamini_hochberg(out.loc[mask, "p_value_group_separation"].tolist())
    return out
