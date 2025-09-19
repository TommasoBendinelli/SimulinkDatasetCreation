# -*- coding: utf-8 -*-
"""
Simulink runner with robust saving + diagnostics for large time series.
"""
import os
import json
from pathlib import Path
from typing import Dict, List
import sys 
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matlab.engine

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from utils import new_run_dir, sanity_check, linear_ramp, logistic_ramp, step

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

def generate_tunable_parameters() -> Dict:
    # Use int() to ensure pure Python scalars (MATLAB Engine is picky)
    res = {
        "Coefficient of Restitution": np.random.uniform(0.6,0.8)
    }

    return res 


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




def compress_df(df, uST, target_column=""):
    breakpoint()
    # Build efficiency profile
    # value = np.full(n_points, initial_value, dtype=float)

    # # Linear ramp from start_val to end_val
    # ramp = np.linspace(initial_value, end_val, drop_end_idx - drop_start_idx + 1)
    # value[drop_start_idx:drop_end_idx + 1] = ramp

    # # After ramp, hold at end_val
    # if drop_end_idx + 1 < n_points:
    #     value[(drop_end_idx + 1):] = end_val
    # # --- Keep only the values that are a change from before ---
    # # Use a small tolerance to avoid emitting duplicate float-equal points
    # atol = 1e-12
    # change_indices = [0]
    # for i in range(1, n_points):
    #     if not np.isclose(value[i], value[i - 1], rtol=0.0, atol=atol):
    #         change_indices.append(i)

    # times_full = np.arange(n_points, dtype=float) * uST
    # time_cp = times_full[change_indices].tolist()
    # value = value[change_indices].tolist()
    # return times_full, time_cp, value

def generate_time_varying_parameters(mle, root_cause=None, uST=0.1, stop_time=30.0, seed=42):
    # Number of points (include t=0 and t=stop_time)
    n_points = int(round(stop_time / uST)) + 1
    parameter = {'identifier': [], 'time': [], 'values': [], 'seen': []}
    # Define parameters to control 
    parameters = ["X0", "Coefficient of Restitution", "Gravitational acceleration"]

    # Fetch the intial values for each of these
    intial_values_dict = {}
    for parameter in parameters:
        identifier = f"simulink_model_original/{parameter}"
        paramInfo = mle.get_param(identifier, 'DialogParameters')
        # Infer the type of block
        if "Value" in paramInfo and "Constant value" in paramInfo["Value"]["Prompt"]:
            initial_value = float(mle.get_param(identifier, 'Value'))
        elif "Gain" in paramInfo:
            initial_value = float(mle.get_param(identifier, 'Gain'))

        intial_values_dict[parameter] = initial_value

    # Randomize slighly variables that can be randomized
    
    np.random.seed(seed)
    intial_values_dict["Coefficient of Restitution"] = np.random.uniform(-0.2,-0.9)
    intial_values_dict["X0"] = np.random.uniform(0.5, 30)

    # Create the pandas dataframe
    times_full = np.arange(n_points, dtype=float) * uST
    df = pd.DataFrame({**{"time": times_full}, **{p: [v]*n_points for p, v in intial_values_dict.items()}})

    # Introduce a fault programmatically 
    faulty_simulation = [{"target_column": "Gravitational acceleration", "values": [3, 20]}]
    entry = random.choice(faulty_simulation)     # pick a random dictionary
    end_value = random.choice(entry["values"]) * np.random.uniform(0.8,1.2)      # pick a random element from "values"
    target_column = entry["target_column"]
    # Sample a random start time
    total_length = (df["time"].max() - df["time"].min())
    start_time = (df["time"].max() - df["time"].min()) * np.random.uniform(0.2,0.8)
    length_ramp = total_length * np.random.uniform(0.01,0.1)
    end_time = start_time + length_ramp
    type_of_corruption = random.choice(["step","linear_ramp","logistic_ramp"])
    if type_of_corruption == "linear_ramp":
        df = linear_ramp(df,target_column=target_column, start_time=start_time, end_time=end_time, end_value=end_value)
    elif type_of_corruption == "logistic_ramp":
        df = logistic_ramp(df,target_column=target_column,start_time=start_time, end_time=end_time, end_value=end_value)
    elif type_of_corruption == "step":
        df = step(df,target_column=target_column,time=start_time,end_value=end_value)


    _, time_cp, value = compress_df(df, uST, target_column=target_column)
    #

    
    parameter["identifier"].append(identifier)
    parameter["time"].append(matlab.double([x for x in time_cp]))
    parameter["values"].append(matlab.double([y for y in value]))
    parameter["seen"].append(matlab.double([0 for _ in value])) # This is used internally by the Matlab script, we need to set all 0 by default

    # ### Air Density Lower
    # identifier = "simulink_model/Heater_Control/Air_Density"
    # initial_value = float(mle.get_param(identifier.replace("simulink_model","simulink_model_original"), 'Gain'))
    # end_val = np.random.uniform(0.0001,0.000001)
    # ramp_duration = np.random.randint(1,20)

    # drop_start_idx, drop_end_idx = ramp_profile(n_points,ramp_duration, uST)
    # _, time_cp, value = compress_df(n_points, initial_value, end_val, drop_end_idx, drop_start_idx, uST)

    # parameter["identifier"].append(identifier)
    # parameter["time"].append(matlab.double([x for x in time_cp]))
    # parameter["values"].append(matlab.double([y for y in value]))
    # parameter["seen"].append(matlab.double([0 for _ in value])) # This is used internally by the Matlab script, we need to set all 0 by default


    # for k in parameter.keys():
    #     parameter[k].pop(1)
    
    return parameter




def generate_time_varying_inputs(root_cause=None, uST=None, stop_time=None, rng_seed=None, seed=None) -> dict:

    # MATLAB-friendly version
    matlab_signals = {}
    return matlab_signals


def generate_data(mle, root_cause=None, uST=None,  stop_time = 10, diagram_dir=None, seed=42):
    # Inputs

    externalInput = generate_time_varying_inputs(root_cause=None, uST=uST, stop_time=stop_time, seed=seed)
    TimeVaryingParameters = generate_time_varying_parameters(mle, root_cause=None, uST=uST, stop_time=stop_time, seed=seed)
    res = mle.sim_the_model("uST",uST, "StopTime", stop_time, "TunableParameters", tunable_params, "ExternalInput", externalInput, "DiagramDataPath", str(diagram_dir), "TimeVaryingParameters", TimeVaryingParameters)

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
        "root_cause": root_cause
    }
    return df, metadata

   
        
# ---------- main ----------
def main():
    seed = 42
    # Optional: make runs reproducible; comment this out if you want pure randomness
    np.random.seed(None)  # or set an int seed
    current_path = os.getcwd()
    os.chdir(Path(__file__).parent)
    print(f"Changed directory to : {Path(__file__).parent}")
    # Start MATLAB engine
    mle = matlab.engine.start_matlab()
    uST = 0.01 


    root_cause = None
    stop_time = 500
    for i in range(1):
        # Use to read default values
        mle.load_system('simulink_model_original.slx')

        # tunable_params = generate_tunable_parameters()
        sanity_check(uST=uST,stop_time=stop_time)
        
        run_dir = new_run_dir(Path(current_path) / "data", system_name=Path(__file__).parent.name, diagram_subdir= "diagram")

        df, metadata= generate_data(mle,root_cause=root_cause, uST=uST, stop_time=stop_time, diagram_dir=run_dir / "diagram", seed=seed)

        
        paths = save_artifacts(run_dir, df, metadata)

        print("Saved artifacts:")
        for k, v in paths.items():
            print(f" - {k}: {v}")

if __name__ == "__main__":
    main()