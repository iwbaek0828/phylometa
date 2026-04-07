from __future__ import annotations

import pandas as pd


def write_result_dict(result: dict, out_file: str):
    df = pd.DataFrame([result])
    if out_file.endswith(".csv"):
        df.to_csv(out_file, index=False)
    else:
        df.to_csv(out_file, sep="\t", index=False)
