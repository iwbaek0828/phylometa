from __future__ import annotations

import numpy as np
import pandas as pd


def patristic_distance_matrix(tree, tip_names: list[str]) -> pd.DataFrame:
    n = len(tip_names)
    mat = np.zeros((n, n), dtype=float)

    for i, a in enumerate(tip_names):
        for j in range(i, n):
            b = tip_names[j]
            d = tree.distance(a, b)
            mat[i, j] = d
            mat[j, i] = d

    return pd.DataFrame(mat, index=tip_names, columns=tip_names)
