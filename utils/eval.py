import pandas as pd
from pathlib import Path

def load_results_df(results_dir: str) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dfs = []
    for f in results_dir.iterdir():
        results_dfs.append(pd.read_pickle(f))
    df = pd.concat(results_dfs, axis=0, ignore_index=True)
    if 'loss' in df.columns:
        df["loss"] = df["loss"].astype(float)
    return df
