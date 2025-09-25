from pathlib import Path 
import time 
import uuid 
import numpy as np
import pandas as pd

def new_run_dir(root: Path = "runs", system_name="entry", diagram_subdir="diagram"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    sid = str(uuid.uuid4())[:8]
    run_dir = root / system_name / f"{ts}__{sid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / diagram_subdir).mkdir(parents=True,exist_ok=False)
    return run_dir

def sanity_check(uST, metadata_dataset=None):
    assert metadata_dataset is not None 
    ### Efficiency 
    if uST is None:
        raise ValueError("uST and stop_time must be provided.")
    if uST <= 0:
        raise ValueError("uST must be > 0.")
    assert 0 < metadata_dataset["time_grid"]["fault_window"]["duration_fraction_range"][0] < metadata_dataset["time_grid"]["fault_window"]["duration_fraction_range"][1]
    assert metadata_dataset["time_grid"]["fault_window"]["duration_fraction_range"][1] < 1

def _value_cols(df: pd.DataFrame):
    return [c for c in df.columns if c != "time"]


def linear_ramp(
    df: pd.DataFrame,
    target_column: str,
    start_time: float,
    end_time: float,
    end_value: float
) -> pd.DataFrame:
    """
    Linearly ramp `target_column` from its value at `start_time` to `end_value`
    over the window [start_time, end_time]. Holds the start value before the
    window and holds `end_value` after the window.

    - If `start_time` is before the first timestamp, the start value is the first value.
    - If `start_time` is after the last timestamp, the start value is the last value.
    - If `start_time == end_time`, this becomes a step to `end_value` at that time.
    """
    assert "time" in df.columns, "DataFrame must contain a 'time' column."
    assert target_column in df.columns, f"Missing column: {target_column}"
    assert end_time >= start_time, "`end_time` must be >= `start_time`."

    out = df.copy()
    t = out["time"].to_numpy()
    y = out[target_column].to_numpy().astype(float)

    t0, tN = t[0], t[-1]

    # Determine start value at start_time (interpolate when inside the range)
    if start_time <= t0:
        start_val = float(y[0])
    elif start_time >= tN:
        start_val = float(y[-1])
    else:
        start_val = float(np.interp(start_time, t, y))

    # Build alpha in [0,1] over the window, with holds outside.
    alpha = np.zeros_like(t, dtype=float)

    if end_time == start_time:
        # Instant step at start_time: before -> 0, at/after -> 1
        alpha = (t >= start_time).astype(float)
    else:
        win = (t >= start_time) & (t <= end_time)
        alpha[win] = (t[win] - start_time) / (end_time - start_time)
        alpha[t > end_time] = 1.0
        # t < start_time stays 0.0

    # Compute ramped values
    ramped = (1.0 - alpha) * start_val + alpha * float(end_value)

    # Write back only the target column
    out.loc[:, target_column] = ramped
    return out


def logistic_ramp(
    df: pd.DataFrame,
    target_column: str,
    start_time: float,
    end_time: float,
    end_value: float,
    k: float = 4.6,
) -> pd.DataFrame:
    """
    Smooth S-shaped (logistic) ramp for `target_column` from its value at `start_time`
    to `end_value` over [start_time, end_time]. Holds before/after.

    - If `start_time` is outside the data's time range, the start value is clamped
      to the first/last value accordingly (consistent with np.interp behavior).
    - If `end_time <= start_time`, this becomes a step at `start_time`.
    """
    assert "time" in df.columns, "DataFrame must contain a 'time' column."
    assert target_column in df.columns, f"Missing column: {target_column}"
    assert target_column != "time", "target_column must not be 'time'."

    out = df.copy()
    t = out["time"].to_numpy()
    y = out[target_column].to_numpy().astype(float)

    # Start value at start_time (interpolated or clamped at ends)
    t0, tN = t[0], t[-1]
    if start_time <= t0:
        start_val = float(y[0])
    elif start_time >= tN:
        start_val = float(y[-1])
    else:
        start_val = float(np.interp(start_time, t, y))

    # Step case: end_time <= start_time
    if end_time <= start_time:
        sigma = (t >= start_time).astype(float)
        ramped = (1.0 - sigma) * start_val + sigma * float(end_value)
        out.loc[:, target_column] = ramped
        return out

    # Logistic alpha that transitions ~[start_time, end_time]
    mid = 0.5 * (start_time + end_time)
    half = max(0.5 * (end_time - start_time), 1e-12)
    tau = (t - mid) / half
    sigma = 1.0 / (1.0 + np.exp(-k * tau))

    # Clamp to perfect hold outside the window
    sigma = np.where(t <= start_time, 0.0, sigma)
    sigma = np.where(t >= end_time, 1.0, sigma)

    # Blend start_val -> end_value
    ramped = (1.0 - sigma) * start_val + sigma * float(end_value)
    out.loc[:, target_column] = ramped
    return out

def step(
    df: pd.DataFrame,
    target_column: str,
    time: float,
    end_value: float
) -> pd.DataFrame:
    """
    Hard step for `target_column`: for t >= `time`, set to `end_value`.
    Holds the original values before `time`. Other columns are unchanged.
    """
    assert "time" in df.columns, "DataFrame must contain a 'time' column."
    assert target_column in df.columns, f"Missing column: {target_column}"
    assert target_column != "time", "target_column must not be 'time'."

    out = df.copy()
    t = out["time"].to_numpy()

    mask = t >= time
    out.loc[mask, target_column] = end_value
    return out