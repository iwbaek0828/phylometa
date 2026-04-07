from Bio import Phylo
from io import StringIO
import pandas as pd
from phylometa.tree import patristic_distance_matrix
from phylometa.stats_categorical import categorical_clustering_test, batch_categorical_tests


def _tree_and_meta():
    tree = Phylo.read(StringIO("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);"), "newick")
    meta = pd.DataFrame({
        "strain": ["A", "B", "C", "D"],
        "habitat": ["soil", "soil", "ice", "lake"],
        "antarctica": ["no", "no", "yes", "no"],
    })
    return tree, meta


def test_categorical_clustering_test_runs():
    tree, meta = _tree_and_meta()
    dist = patristic_distance_matrix(tree, ["A", "B", "C", "D"])
    out = categorical_clustering_test(dist, meta, "strain", "habitat", n_perm=20, seed=1)
    assert "p_value_clustering" in out
    assert 0 <= out["p_value_clustering"] <= 1


def test_batch_categorical_tests_adds_qvalues():
    tree, meta = _tree_and_meta()
    dist = patristic_distance_matrix(tree, ["A", "B", "C", "D"])
    out = batch_categorical_tests(dist, meta, "strain", ["habitat", "antarctica"], n_perm=20, seed=1)
    assert "q_value_clustering" in out.columns
