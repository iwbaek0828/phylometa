import pandas as pd
from phylometa.qc import check_tip_metadata_overlap


def test_check_tip_metadata_overlap():
    tips = ["A", "B", "C"]
    meta = pd.DataFrame({"strain": ["A", "B", "D"]})
    out = check_tip_metadata_overlap(tips, meta, "strain")
    assert out["n_shared"] == 2
    assert out["only_tree"] == ["C"]
    assert out["only_metadata"] == ["D"]
