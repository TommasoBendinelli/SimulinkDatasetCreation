# -*- coding: utf-8 -*-
"""
Simulink runner with robust saving + diagnostics for large time series.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys 
import shutil
import json

   
import click
from pathlib import Path

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
              if not name in ["OperatingPoint"] or "Signal_" in name]
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
    csv_path = run_dir / "data.csv"
    df["time"] = df.index
    # Reset index
    df = df.reset_index(drop=True)
    df.to_csv(csv_path,  index=False)

    # Just for the visualization
    df.index = df["time"]
    df.drop("time",inplace=True, axis=1)

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
    meta_path = run_dir / "metadata_task.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "csv": str(csv_path),
        "preview": str(preview_path),
        "metadata": str(meta_path),
    }




def compress_df(df, target_column=""):
    # # Use a small tolerance to avoid emitting duplicate float-equal points
    atol = 1e-12
    change_indices = [0]
    values = df[target_column]
    for i in range(1, len(df)):
        if not np.isclose(values[i], values[i - 1], rtol=0.0, atol=atol):
            change_indices.append(i)
    time_delta = df["time"][change_indices].values
    values_delta = values[change_indices].values
    return time_delta, values_delta

def sample_value(dict_metadata):
    if dict_metadata["type"] == "uniform":
        value = np.random.uniform(dict_metadata['min'], dict_metadata['max'])
    else:
        breakpoint()
        raise NotImplementedError()
    return value

def generate_time_varying_parameters(mle, uST=0.1, stop_time=30.0, metadata=None) -> Tuple[dict,dict]:
    # Number of points (include t=0 and t=stop_time)
    n_points = int(round(stop_time / uST)) + 1
    res_dict = {'identifier': [], 'time': [], 'values': [], 'seen': [], 'key': []}
    # Define parameters to control 
    blocks_type = {}
    # Fetch the intial values for each of these
    intial_values_dict = {}
    for parameter in metadata['parameters']:
        # initial_value = float(mle.get_param(parameter['identifier'], parameter['key']))
        intial_value = sample_value(parameter["initial_sampling"])
        intial_values_dict[parameter['identifier']] = intial_value

        blocks_type[parameter['identifier']] = parameter['key']

    # Create the pandas dataframe
    times_full = np.arange(n_points, dtype=float) * uST
    df = pd.DataFrame({**{"time": times_full}, **{p: [v]*n_points for p, v in intial_values_dict.items()}})
    
    # Metadata options
    possible_errors = [x for x in metadata['fault_options'] if ("fault" not in x) or (x["fault"] is not None)]
    
    # Introduce a fault programmatically # 
    # faulty_simulation = [  {"target_column": "Gravitational acceleration", "values": [-3, -20, 2]},  {"target_column":"Coefficient of Restitution","values": [sampled_coefficient - 0.5]}]
    error = random.choice(possible_errors)     # pick a random dictionary
    root_cause = {}
    if "fault" in error:
        sample_strategy = random.choice(error['fault']['value_candidates'])
        end_value = sample_value(sample_strategy)
        target_column = error['fault']["parameter"]
        # Sample a random start time
        start_fault_window = metadata["time_grid"]["fault_window"]["start_fraction_range"][0]
        end_fault_window = metadata["time_grid"]["fault_window"]["start_fraction_range"][1]
        duration_fract_min = metadata["time_grid"]["fault_window"]["duration_fraction_range"][0]
        duration_fract_max = metadata["time_grid"]["fault_window"]["duration_fraction_range"][1]

        total_length = (df["time"].max() - df["time"].min())
        start_time = np.random.uniform(start_fault_window,end_fault_window)
        duration_length = total_length * np.random.uniform(duration_fract_min,duration_fract_max)
        end_time = start_time + duration_length
        available_corruptions = error['fault']['allowed_corruption_types']
        type_of_corruption = random.choice(available_corruptions)
        
        # --- DEBUG PRINTS ---
        print("\n--- Fault Injection Details ---")
        print(f"Target column        : {target_column}")
        print(f"Sample strategy      : {sample_strategy}")
        print(f"Chosen end value     : {end_value}")
        print(f"Available corruptions: {available_corruptions}")
        print(f"Selected corruption  : {type_of_corruption}")
        print(f"Start fault window   : {start_fault_window}")
        print(f"End fault window     : {end_fault_window}")
        print(f"Duration fraction rng: ({duration_fract_min}, {duration_fract_max})")
        print(f"Total signal length  : {total_length}")
        print(f"Start time           : {start_time}")
        print(f"Duration length      : {duration_length}")
        print(f"End time             : {end_time}")
        print("-------------------------------\n")
        if type_of_corruption == "linear_ramp":
            df = linear_ramp(df,target_column=target_column, start_time=start_time, end_time=end_time, end_value=end_value)
        elif type_of_corruption == "logistic_ramp":
            df = logistic_ramp(df,target_column=target_column,start_time=start_time, end_time=end_time, end_value=end_value)
        elif type_of_corruption == "step":
            df = step(df,target_column=target_column,time=start_time,end_value=end_value)

        time_delta, values_delta = compress_df(df, target_column=target_column)
        res_dict["identifier"].append(target_column.replace("simulink_model_original","simulink_model"))
        res_dict["time"].append(matlab.double([x for x in time_delta]))
        res_dict["values"].append(matlab.double([y for y in values_delta]))
        res_dict["seen"].append(matlab.double([0 for _ in values_delta])) # This is used internally by the Matlab script, we need to set all 0 by default
        res_dict["key"].append(blocks_type[target_column])
        root_cause["root_cause"] = target_column
        root_cause["starting_time"] = start_time
        root_cause["ending_time"] = end_time
        root_cause["new_value"] = end_value
        root_cause["transition_type"] = type_of_corruption
        root_cause["correct_answer"] = error["text"]
    else:
        root_cause["correct_answer"] = error["text"]
        res_dict = None
        
    return res_dict, root_cause
    




def generate_time_varying_inputs(root_cause=None, uST=None, stop_time=None, rng_seed=None, metadata=None) -> Optional[dict]:
    return None

def run_simulation(
    mle,
    *,
    uST,
    stop_time,
    diagram_dir,
    time_varying_parameters=None,
    external_input=None,
    debug=True
):
        """Run the model simulation with optional inputs."""
        if uST is None or stop_time is None or diagram_dir is None:
            raise ValueError("uST, stop_time, and diagram_dir are required.")

        args = [
            "uST", uST,
            "StopTime", stop_time,
            "DiagramDataPath", str(diagram_dir),
            "debug", debug
        ]

        if time_varying_parameters is not None:
            args += ["TimeVaryingParameters", time_varying_parameters]

        if external_input is not None:
            args += ["ExternalInput", external_input]
        
        res =  mle.sim_the_model(*args)
        mle.close_system('simulink_model', 0, nargout=0)
        return res 

def generate_data(mle, uST=None, diagram_dir=None,seed=None):
    metadata_dataset = json.loads(Path("metadata.json").read_text(encoding="utf-8"))
    # Inputs
    sanity_check(uST=uST, metadata_dataset=metadata_dataset)
    stop_time = metadata_dataset["time_grid"]["stop_time"]

    external_input = generate_time_varying_inputs(root_cause=None, uST=uST, stop_time=stop_time, metadata=metadata_dataset)
    time_varying_parameters, root_cause = generate_time_varying_parameters(mle, uST=uST, stop_time=stop_time, metadata=metadata_dataset)
    shutil.copy( "simulink_model_original.slx", "simulink_model.slx")
    print("Running simulation with fault")
    
    res_broken = run_simulation(mle=mle, uST=uST, stop_time=stop_time, diagram_dir=diagram_dir, time_varying_parameters=time_varying_parameters, external_input=external_input, debug=False)
    shutil.copy("simulink_model_original.slx", "simulink_model.slx")
    # Make a copy of time_varying_parameters
    if time_varying_parameters:
        time_varying_parameters_initial = time_varying_parameters.copy()
        for key in time_varying_parameters_initial.keys():
            if key in ["identifier","key"]:
                continue
            else:
                new = []
                for entry in time_varying_parameters_initial[key]:
                    new.append(matlab.double(entry[0][0]))
                time_varying_parameters_initial[key] = new
    else:
        time_varying_parameters_initial = None
    print("Running simulations with no faults")
    res_healthy = run_simulation(mle=mle, uST=uST, stop_time=stop_time, diagram_dir=diagram_dir, time_varying_parameters=time_varying_parameters_initial, external_input=external_input, debug=False)

    # Convert MATLAB results to DataFrame
    df_broken = get_time_series(res_broken, assuming_all_scalar=True)
    df_healthy = get_time_series(res_healthy, assuming_all_scalar=True)

    # Check if the two are equal
    # Compare DataFrames with tolerance
    # Reindex both DataFrames to the union of their indices
    common_index = df_broken.index.union(df_healthy.index)

    df_broken_interp = df_broken.reindex(common_index).interpolate(method="linear")
    df_healthy_interp = df_healthy.reindex(common_index).interpolate(method="linear")

    # Compare with tolerance (result is a numpy array)
    comparison = np.isclose(df_broken_interp, df_healthy_interp, atol=0.1, equal_nan=True)
    # Convert back to DataFrame with same index/columns
    comparison = pd.DataFrame(comparison, index=common_index, columns=df_broken.columns)

    # Find rows where at least one column differs beyond tolerance
    diff_rows = ~comparison.all(axis=1)

    # Get the first index where they differ
    first_diff_index = float(diff_rows.idxmax()) if diff_rows.any() else None
    is_observerd = {"valid":bool(diff_rows.any()), "first_diff_time": first_diff_index}
    # Prepare metadata describing the run (super handy for digging later)
    metadata = {
        "model": "the_model",
        "stop_time_s": stop_time,
        "n_rows": int(df_broken.shape[0]),
        "n_signals": int(df_broken.shape[1]) if not df_broken.empty else 0,
        "time_start_s": float(df_broken.index.min()) if not df_broken.empty else None,
        "time_end_s": float(df_broken.index.max()) if not df_broken.empty else None,
        "columns": list(df_broken.columns) if not df_broken.empty else [],
        "seed": seed,
        "env": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
            # Add MATLAB release if you like:
            # "matlab_release": str(mle.version(nargout=1))
        },
    }

    # Create the prompt for the model
    prompt_related = {}
    prompt_related["context"] = metadata_dataset["context"] 
    prompt_related["question"] = metadata_dataset["question"]
    prompt_related["options"] = [x["text"] for x in metadata_dataset["fault_options"]]
    metadata = {**metadata,**root_cause, **is_observerd, **prompt_related}
    return df_broken, metadata

@click.command()
@click.argument(
    "index",
    type=click.IntRange(0, 1),
)
def main(index):
    available_scenarios = ["BouncingBall", "MassSpringDamperWithController"]
    root_path = Path("models") / available_scenarios[index]
    click.echo(f"Using scenario: {available_scenarios[index]} at {root_path}")
    cwd = os.getcwd()
    os.chdir(Path(cwd) / root_path)
    mle = matlab.engine.start_matlab()
    uST = 0.01 

    

    # metadata_path = Path("metadata.json")
    for i in range(1):
        random.seed(i)
        np.random.seed(i)
        # Make a copy of simulink_model_original

        # Use to read default values
        mle.load_system(str(root_path/ 'simulink_model_original.slx'))
        

       
        run_dir = new_run_dir(Path(cwd) / "data", system_name=root_path.name, diagram_subdir= "diagram")

        df, metadata= generate_data(mle, uST=uST, diagram_dir=run_dir / "diagram", seed=i)

        paths = save_artifacts(run_dir, df, metadata)

        print("Saved artifacts:")
        for k, v in paths.items():
            print(f" - {k}: {v}")

if __name__ == "__main__":
    main()