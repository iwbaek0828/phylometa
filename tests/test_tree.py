from Bio import Phylo
from io import StringIO
from phylometa.tree import patristic_distance_matrix


def test_patristic_distance_matrix_shape():
    tree = Phylo.read(StringIO("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);"), "newick")
    tips = ["A", "B", "C", "D"]
    dist = patristic_distance_matrix(tree, tips)
    assert dist.shape == (4, 4)
    assert dist.loc["A", "A"] == 0
