from Bio import Phylo
from io import StringIO
import pandas as pd
from phylometa.stats_categorical import clade_enrichment_test


def test_clade_enrichment_test_runs():
    tree = Phylo.read(StringIO("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);"), "newick")
    meta = pd.DataFrame({
        "strain": ["A", "B", "C", "D"],
        "antarctica": ["no", "no", "yes", "yes"],
    })
    out = clade_enrichment_test(tree, meta, "strain", "antarctica", "yes", min_clade_size=2)
    assert "q_value" in out.columns
