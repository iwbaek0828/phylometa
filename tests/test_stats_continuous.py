from Bio import Phylo
from io import StringIO
import pandas as pd
from phylometa.tree import patristic_distance_matrix
from phylometa.stats_continuous import continuous_trait_association_test


def test_continuous_trait_association_test_runs():
    tree = Phylo.read(StringIO("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);"), "newick")
    meta = pd.DataFrame({
        "strain": ["A", "B", "C", "D"],
        "ros_burden": [11, 13, 22, 14],
    })
    dist = patristic_distance_matrix(tree, ["A", "B", "C", "D"])
    out = continuous_trait_association_test(dist, meta, "strain", "ros_burden", n_perm=20, seed=1)
    assert "p_value" in out
    assert 0 <= out["p_value"] <= 1
