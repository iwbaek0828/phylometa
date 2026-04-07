from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def continuous_trait_association_test(
    dist_df: pd.DataFrame,
    meta: pd.DataFrame,
    id_column: str,
    trait: str,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    if trait not in meta.columns:
        raise ValueError(f"Trait column '{trait}' not found.")

    df = meta[[id_column, trait]].dropna().copy()
    df[id_column] = df[id_column].astype(str)
    df = df[df[id_column].isin(dist_df.index)].copy()

    if df.shape[0] < 3:
        raise ValueError("At least 3 samples are required for continuous trait testing.")

    df = df.set_index(id_column)
    ids = df.index.tolist()
    trait_values = df[trait].astype(float)

    subdist = dist_df.loc[ids, ids].values
    trait_diff = np.abs(trait_values.values[:, None] - trait_values.values[None, :])

    iu = np.triu_indices_from(subdist, k=1)
    x = subdist[iu]
    y = trait_diff[iu]

    observed_corr, _ = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    perm_corrs = []
    base = trait_values.to_numpy().copy()

    for _ in range(n_perm):
        shuffled = base.copy()
        rng.shuffle(shuffled)
        shuffled_diff = np.abs(shuffled[:, None] - shuffled[None, :])
        y_perm = shuffled_diff[iu]
        corr, _ = spearmanr(x, y_perm)
        perm_corrs.append(corr)

    perm_corrs = np.array(perm_corrs, dtype=float)
    p_value = (np.sum(np.abs(perm_corrs) >= abs(observed_corr)) + 1) / (len(perm_corrs) + 1)

    return {
        "trait": trait,
        "n_samples": int(len(ids)),
        "spearman_rho": float(observed_corr),
        "permuted_mean_rho": float(np.nanmean(perm_corrs)),
        "p_value": float(p_value),
        "interpretation": "Positive rho means phylogenetically distant samples tend to differ more in trait value.",
    }
