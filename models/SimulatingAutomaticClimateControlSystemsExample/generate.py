# -*- coding: utf-8 -*-
"""
Simulink runner with robust saving + diagnostics for large time series.
"""
import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matlab.engine

# ---------- helpers ----------
def _to1d(a):
    # works for matlab.double([[...],[...]]) and plain arrays
    return np.asarray(a).reshape(-1)

def block_to_series(name: str, block) -> pd.Series:
    t = _to1d(block["Time"]).astype(float)
    y = _to1d(block["Data"]).astype(float)
    s = pd.Series(y, index=t, name=name)
    s.index.name = "time_s"
    # collapse exact-duplicate timestamps; keep mean (use .last() if preferred)
    s = s.groupby(level=0).mean().sort_index()
    return s

def get_time_series(res: Dict, assuming_all_scalar: bool = True) -> pd.DataFrame:
    if not assuming_all_scalar:
        raise NotImplementedError("Vector/matrix signals not handled yet.")
    # build one Series per (non-temporary) key
    series = [block_to_series(name, block)
              for name, block in res.items()
              if "Signal_" not in name]
    if not series:
        return pd.DataFrame()
    # outer-join on the union of all time points
    df = pd.concat(series, axis=1).sort_index()
    # forward-fill to align columns; drop any all-NaN leading rows
    df = df.ffill().dropna(how="all")
    # enforce dtype + tidy index
    df.index = df.index.astype("float64")
    df.index.name = "time_s"
    return df

def generate_random_healthy_variables() -> Dict:
    # Use int() to ensure pure Python scalars (MATLAB Engine is picky)
    return {
        "u_set_point": float(int(np.random.randint(18, 24))),                 # [°C]
        "u_external_temperature": float(int(np.random.randint(-10, 40))),     # [°C]
        "u_recycling_air_bool": int(np.random.randint(0, 2)),                 # 0 or 1
        "u_engine_speed_int": int(np.random.randint(1500, 2500)),             # [rpm]
        "u_torque_compensation": int(np.random.randint(20, 30)),
        "param_power_per_occupants": int(np.random.randint(50, 150)),
        "param_first_order_inertia": float(1.0 / int(np.random.randint(2500, 5000))),
    }

def new_run_dir(root: Path = "runs", system_name="entry"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    sid = str(uuid.uuid4())[:8]
    run_dir = root / system_name / f"{ts}__{sid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def save_artifacts(run_dir: Path, df: pd.DataFrame, metadata: Dict, preview_cols: int = 6):
    # 1) primary data (Parquet)
    parquet_path = run_dir / "data.parquet"
    df.to_parquet(parquet_path)  # requires pyarrow or fastparquet

    # 2) tiny CSV head for quick eyeballing
    head_path = run_dir / "data_head.csv.gz"
    df.iloc[:100].to_csv(head_path, index=True, compression="gzip")

    # 3) quick plot preview (first few columns)
    plt.figure()
    if not df.empty:
        df.iloc[:, :min(preview_cols, df.shape[1])].plot(legend=True)
    plt.xlabel("time [s]")
    plt.tight_layout()
    preview_path = run_dir / "preview.png"
    plt.savefig(preview_path, dpi=150)
    plt.close()

    # 4) metadata sidecar (human-friendly)
    meta_path = run_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "parquet": str(parquet_path),
        "csv_head": str(head_path),
        "preview": str(preview_path),
        "metadata": str(meta_path),
    }

# ---------- main ----------
def main():
    # Optional: make runs reproducible; comment this out if you want pure randomness
    np.random.seed(None)  # or set an int seed
    current_path = os.getcwd()
    os.chdir(Path(__file__).parent)

    # Start MATLAB engine
    mle = matlab.engine.start_matlab()

    for i in range(10):
        # Inputs
        tunable_params = generate_random_healthy_variables()
        stop_time = 1000  # seconds

        # Run simulation (adapt function name/args to your setup)
        res = mle.sim_the_model("StopTime", stop_time, "TunableParameters", tunable_params)

        # Convert MATLAB results to DataFrame
        df = get_time_series(res, assuming_all_scalar=True)

        # Prepare metadata describing the run (super handy for digging later)
        metadata = {
            "model": "the_model",
            "stop_time_s": stop_time,
            "n_rows": int(df.shape[0]),
            "n_signals": int(df.shape[1]) if not df.empty else 0,
            "time_start_s": float(df.index.min()) if not df.empty else None,
            "time_end_s": float(df.index.max()) if not df.empty else None,
            "columns": list(df.columns) if not df.empty else [],
            "tunable_parameters": tunable_params,
            "env": {
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "pandas": pd.__version__,
                "matplotlib": plt.matplotlib.__version__,
                # Add MATLAB release if you like:
                # "matlab_release": str(mle.version(nargout=1))
            },
        }

        # Save everything under a timestamped run folder
        run_dir = new_run_dir(Path(current_path) / "data", system_name=Path(__file__).parent.name)
        paths = save_artifacts(run_dir, df, metadata)

        print("Saved artifacts:")
        for k, v in paths.items():
            print(f" - {k}: {v}")

if __name__ == "__main__":
    main()