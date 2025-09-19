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

def generate_tunable_parameters(root_cause=None) -> Dict:
    # Use int() to ensure pure Python scalars (MATLAB Engine is picky)
    res = {
        "u_recycling_air_bool": int(np.random.randint(0, 2)),                 # 0 or 1
        "u_engine_speed_int": int(np.random.randint(1500, 2500)),             # [rpm]
        "u_torque_compensation": int(np.random.randint(20, 30)),
        "param_power_per_occupants": int(np.random.randint(50, 150)),
        "param_first_order_inertia": float(1.0 / int(np.random.randint(2500, 5000))),
    }

    return res 

def new_run_dir(root: Path = "runs", system_name="entry", diagram_subdir="diagram"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    sid = str(uuid.uuid4())[:8]
    run_dir = root / system_name / f"{ts}__{sid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / diagram_subdir).mkdir(parents=True,exist_ok=False)
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

def ramp_profile(n_points,ramp_duration, uST):

    # If stop_time < desired_drop_seconds, clamp the drop to what's possible
    drop_len = ramp_duration / uST

    # Latest index the drop can start so it finishes by the end
    latest_start_idx = max(0, n_points - drop_len)
    # Deterministic RNG for reproducibility
    rng_seed = 4
    rng = np.random.default_rng(rng_seed)

    # Inclusive of latest_start_idx so the ramp can end exactly at the last sample
    drop_start_idx = int(rng.integers(0, latest_start_idx + 1))
    drop_end_idx = int(min(n_points - 1, drop_start_idx + drop_len - 1))
    return drop_start_idx, drop_end_idx

def modify_value(n_points, initial_value, end_val, drop_end_idx, drop_start_idx, uST):
    # Build efficiency profile
    value = np.full(n_points, initial_value, dtype=float)

    # Linear ramp from start_val to end_val
    ramp = np.linspace(initial_value, end_val, drop_end_idx - drop_start_idx + 1)
    value[drop_start_idx:drop_end_idx + 1] = ramp

    # After ramp, hold at end_val
    if drop_end_idx + 1 < n_points:
        value[(drop_end_idx + 1):] = end_val
    # --- Keep only the values that are a change from before ---
    # Use a small tolerance to avoid emitting duplicate float-equal points
    atol = 1e-12
    change_indices = [0]
    for i in range(1, n_points):
        if not np.isclose(value[i], value[i - 1], rtol=0.0, atol=atol):
            change_indices.append(i)

    times_full = np.arange(n_points, dtype=float) * uST
    time_cp = times_full[change_indices].tolist()
    value = value[change_indices].tolist()
    return times_full, time_cp, value

def generate_time_varying_parameters(mle, root_cause=None, uST=0.1, stop_time=30.0):
    """
    Generate a time series for 'Error Efficiency' that linearly drops from 0.86 to 0.05
    over ~10 s starting at a random time, then holds. Only change-points are returned.

    Parameters
    ----------
    root_cause : any, optional
        Arbitrary metadata to carry through (not used in the generation logic).
    uST : float
        Sample period in seconds (must be > 0).
    stop_time : float
        Total duration in seconds (must be > 0).

    Returns
    -------
    dict
        {
          'time': list[float],          # timestamps at change-points only
          'efficiency': list[float],    # values at those timestamps
          'drop_start_time': float,     # when the ramp starts
          'drop_end_time': float,       # when the ramp ends
          'root_cause': any             # passthrough of input
        }
    """

    entry = {'identifier': [], 'time': [], 'values': [], 'seen': []}
    # Number of points (include t=0 and t=stop_time)
    n_points = int(round(stop_time / uST)) + 1

    ### Efficiency 
    if uST is None or stop_time is None:
        raise ValueError("uST and stop_time must be provided.")
    if uST <= 0:
        raise ValueError("uST must be > 0.")
    if stop_time <= 0:
        raise ValueError("stop_time must be > 0.")

    # Error Efficiency: 0.86 → 0.05 over ~10 s at a random onset time
    identifier = "simulink_model/AC_Control/Efficiency"
    initial_value = float(mle.get_param(identifier.replace("simulink_model","simulink_model_original"), 'Gain'))

    end_val = np.random.uniform(0.01,0.05)
    ramp_duration = np.random.randint(1,20)

    drop_start_idx, drop_end_idx = ramp_profile(n_points,ramp_duration, uST)

    _, time_cp, value = modify_value(n_points, initial_value, end_val, drop_end_idx, drop_start_idx, uST)
    #
    entry["identifier"].append(identifier)
    entry["time"].append(matlab.double([x for x in time_cp]))
    entry["values"].append(matlab.double([y for y in value]))
    entry["seen"].append(matlab.double([0 for _ in value])) # This is used internally by the Matlab script, we need to set all 0 by default

    ### Air Density Lower
    identifier = "simulink_model/Heater_Control/Air_Density"
    initial_value = float(mle.get_param(identifier.replace("simulink_model","simulink_model_original"), 'Gain'))
    end_val = np.random.uniform(0.0001,0.000001)
    ramp_duration = np.random.randint(1,20)

    drop_start_idx, drop_end_idx = ramp_profile(n_points,ramp_duration, uST)
    _, time_cp, value = modify_value(n_points, initial_value, end_val, drop_end_idx, drop_start_idx, uST)

    entry["identifier"].append(identifier)
    entry["time"].append(matlab.double([x for x in time_cp]))
    entry["values"].append(matlab.double([y for y in value]))
    entry["seen"].append(matlab.double([0 for _ in value])) # This is used internally by the Matlab script, we need to set all 0 by default


    for k in entry.keys():
        entry[k].pop(1)
    
    return entry




def generate_time_varying_inputs(root_cause=None, uST=None, stop_time=None, rng_seed=None):
    if uST is None or stop_time is None:
        raise ValueError("uST and stop_time must be provided.")
    if uST <= 0 or stop_time <= 0:
        raise ValueError("uST and stop_time must be positive.")

    # Number of points (include t=0 and t=stop_time)
    n_points = int(stop_time / uST) + 1
    t = np.linspace(0.0, stop_time, n_points)

    half_time = int(np.random.uniform(len(t)*0.1,len(t)*0.9))
    t_half = t[half_time]
    u1 = np.where(t > t_half, 21.0, 23.0)

    # u2: triangular profile (28 → 35 → 27)

    # summer 
    if False:
        third = n_points // 3
        start = 18 
        high_temperature = 35 
        u2_first  = np.full(third, 28.0)
        u2_second = np.linspace(28.0, high_temperature, third, endpoint=False)
        u2_third  = np.linspace(high_temperature, 27.0, n_points - 2 * third)
        u2 = np.concatenate([u2_first, u2_second, u2_third])

    # winter
    if True:
        third = n_points // 3
        start = -5
        high_temperature = 30
        u2_first  = np.full(third, -5)
        u2_second = np.linspace(start, high_temperature, third, endpoint=False)
        u2_third  = np.linspace(high_temperature,  -5,  n_points - 2 * third)
        u2 = np.concatenate([u2_first, u2_second, u2_third])


    # Pack into dict
    u1 = u1 + 273
    u2 = u2 + 273
    signals = {
        "InputSetTemperature": u1.tolist(),
        "ExternalTemperature": u2.tolist(),
    }

    # MATLAB-friendly version
    matlab_signals = {k: matlab.double(np.atleast_2d(v).tolist()) for k, v in signals.items()}
    return matlab_signals


def generate_data(mle, root_cause=None, uST=None,  stop_time = 10, diagram_dir=None):
    # Inputs

    tunable_params = generate_tunable_parameters(root_cause=root_cause)
    externalInput = generate_time_varying_inputs(root_cause=None, uST=uST, stop_time=stop_time)
    TimeVaryingParameters = generate_time_varying_parameters(mle, root_cause=None, uST=uST, stop_time=stop_time)
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
        # mle.set_param('simulink_model', 'SimulationCommand', 'start')

        # Save everything under a timestamped run folder
        run_dir = new_run_dir(Path(current_path) / "data", system_name=Path(__file__).parent.name, diagram_subdir= "diagram")

        df, metadata= generate_data(mle,root_cause=root_cause, uST=uST, stop_time=stop_time, diagram_dir=run_dir / "diagram")

        
        paths = save_artifacts(run_dir, df, metadata)

        print("Saved artifacts:")
        for k, v in paths.items():
            print(f" - {k}: {v}")

if __name__ == "__main__":
    main()