# -*- coding: utf-8 -*-
"""
Example showing how to simulate a Simulink model (called the_model) with different
parameter and external input signal values using the MATLAB Engine API for Python.
"""
import numpy as np
import matlab.engine
import pandas as pd
import matplotlib.pyplot as plt
from typing import List 
import os
from pathlib import Path 

# --- helper: matlab.double -> 1D numpy array ---
def _to1d(a):
    # works for matlab.double([[...],[...]]) and plain arrays
    return np.asarray(a).reshape(-1)

def block_to_series(name, block):
    t = _to1d(block['Time'])
    y = _to1d(block['Data'])
    s = pd.Series(y, index=t, name=name)
    # collapse exact-duplicate timestamps (keep mean; you can use .last() instead)
    s = s.groupby(level=0).mean().sort_index()
    return s


def get_time_series(res0: List,  assuming_all_scalar=True):
    if not assuming_all_scalar:
        raise NotImplementedError("Error")
    # build one Series per key
    series = [block_to_series(name, block) for name, block in res0.items() if not "Signal_" in name]


    # outer-join on the UNION of all time points
    df = pd.concat(series, axis=1).sort_index()

    # df now has time (seconds) as the index and one column per key
    print(df.head())

    return df

def main():
    os.chdir(Path(__file__).parent)
    # Start MATLAB engine
    mle = matlab.engine.start_matlab()
    # breakpoint()
    # mle.sim_the_model()
    # Change MATLAB's current directory
    # mle.addpath(r'IndustrialRootAnalysisBench/models/SimulatingAutomaticClimateControlSystemsExample', nargout=0)

    # Run second simulation (modified tunable parameters)
    tunableParams = {
        'u_set_point': 18.0,
        'u_external_temperature':  39.0,
        'u_recycling_air_bool': 1,
        'u_recycling_air_bool': 1,
        'u_recycling_air_bool': 200,
        'u_torque_compensation': 25
    }


    # u = np.concatenate([np.zeros(1), 2*np.ones(1), np.zeros(3), -100*np.ones(2), np.ones(1)])[None,:]
    # u2 = np.concatenate([np.zeros(1), 2*np.zeros(1), 100*np.ones(3), -2*np.zeros(2), np.zeros(1)])[None,:]
    # u_final = np.concatenate([u,u2],axis=0)
    stop_time = 10

    # FIXED: Correct structure format for ExternalInput
    # This matches the expected From Workspace format in Simulink
    #externalInput = matlab.double(u_final.tolist())
    res1 = mle.sim_the_model('StopTime', stop_time, 'TunableParameters', tunableParams)
    # # Convert MATLAB results to pandas DataFrames
    # df_0 = pd.DataFrame(res0)
    # df_1 = pd.DataFrame(res1)

    # time, df = get_time_series(res0, assuming_all_scalar=True)
    df = get_time_series(res1, assuming_all_scalar=True)

    # Plot signals:
    df.plot(legend=True)
    plt.xlabel("time [s]")
    plt.tight_layout()
    plt.savefig("entry_2.png")

if __name__ == "__main__":
    main()