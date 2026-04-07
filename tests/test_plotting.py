from pathlib import Path
from Bio import Phylo
from io import StringIO
import pandas as pd
from phylometa.plotting import plot_tree_with_traits


def test_plot_tree_with_traits_creates_file(tmp_path: Path):
    tree = Phylo.read(StringIO("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);"), "newick")
    meta = pd.DataFrame({
        "strain": ["A", "B", "C", "D"],
        "habitat": ["soil", "soil", "ice", "lake"],
        "ros_burden": [11, 13, 22, 14],
    })
    out = tmp_path / "plot.pdf"
    plot_tree_with_traits(tree, meta, "strain", ["habitat", "ros_burden"], str(out))
    assert out.exists()
