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
    series = [block_to_series(name, block) for name, block in res0.items()]


    # outer-join on the UNION of all time points
    df = pd.concat(series, axis=1).sort_index()

    # df now has time (seconds) as the index and one column per key
    print(df.head())
    # breakpoint()

    # # Flatten nested lists/arrays
    # df = pd.DataFrame(res0)
    # df = df.applymap(lambda x: np.array(x).flatten() if isinstance(x, list) else x)

    # # Check if time is the same for each variable 
    # times = [df[x]["Time"] for x in df.keys()]
    # assert [times[0] == times[i] for i,x  in enumerate(times)]
    # time = [x[0] for x in times[0]]
    # output_variables = {k: df[k]["Data"] for k in df.keys() if len(df[k]["Data"] ) == len(time)}
    # if assuming_all_scalar:
    #     for k in output_variables.keys():
    #         output_variables[k] = [x[0] for x in output_variables[k]]
    # df = pd.DataFrame(output_variables)
    # df["time"] = time
    return df


# Start MATLAB engine
mle = matlab.engine.start_matlab()

# Change MATLAB's current directory
mle.addpath(r'models/ModelingAFaultTolerantFuelControlSystemExample', nargout=0)


# Run first simulation (default parameters)
res0 = mle.sim_the_model()

# Run second simulation (modified tunable parameters)
tunableParams = {
    'dx2min': -3.0,
    'dx2max':  4.0,
    'gh': 1
}


u = np.concatenate([np.zeros(1), 2*np.ones(1), np.zeros(3), -100*np.ones(2), np.ones(1)])[None,:]
u2 = np.concatenate([np.zeros(1), 2*np.zeros(1), 100*np.ones(3), -2*np.zeros(2), np.zeros(1)])[None,:]
u_final = np.concatenate([u,u2],axis=0)
stop_time = 50

# FIXED: Correct structure format for ExternalInput
# This matches the expected From Workspace format in Simulink
externalInput = matlab.double(u_final.tolist())
res1 = mle.sim_the_model('StopTime', stop_time, 'TunableParameters', tunableParams, 'ExternalInput', externalInput)

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