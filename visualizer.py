# app.py
# Minimal Streamlit visualizer for a SINGLE run folder:
# <run_path>/{data.parquet, metadata.json, data_head.csv.gz, preview.png}
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Run Visualizer", layout="wide")

# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def load_run(run_path: str):
    """Load df + metadata from a single run path."""
    p = Path(run_path)
    if not p.exists() or not p.is_dir():
        return pd.DataFrame(), {}, f"Path not found or not a directory: {run_path}"

    # metadata
    meta = {}
    mp = p / "metadata.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Failed to parse metadata.json: {e}")

    # data
    df: Optional[pd.DataFrame] = None
    pq = p / "data.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            st.warning(f"Failed to read Parquet ({pq}): {e}")

    if df is None:
        csv_head = p / "data_head.csv.gz"
        if csv_head.exists():
            try:
                df = pd.read_csv(csv_head, index_col=0)
            except Exception as e:
                st.warning(f"Failed to read CSV head ({csv_head}): {e}")

    if df is None:
        return pd.DataFrame(), meta, "No readable data file found (data.parquet or data_head.csv.gz)."

    # index hygiene
    if df.index.name is None:
        df.index.name = "time_s"
    try:
        df.index = df.index.astype(float)
    except Exception:
        pass
    df = df.sort_index()
    return df, meta, None


def filter_and_resample(
    df: pd.DataFrame,
    t_min: Optional[float],
    t_max: Optional[float],
    resample_rule: Optional[str],
    agg: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    # time slice on numeric seconds index
    if t_min is not None:
        df = df[df.index >= t_min]
    if t_max is not None:
        df = df[df.index <= t_max]
    if df.empty or not resample_rule:
        return df

    # resample using TimedeltaIndex to avoid tz issues
    ti = pd.to_timedelta(df.index, unit="s")
    tmp = df.copy()
    tmp.index = ti
    if agg == "mean":
        out = tmp.resample(resample_rule).mean()
    elif agg == "median":
        out = tmp.resample(resample_rule).median()
    elif agg == "last":
        out = tmp.resample(resample_rule).last()
    elif agg == "first":
        out = tmp.resample(resample_rule).first()
    elif agg == "max":
        out = tmp.resample(resample_rule).max()
    elif agg == "min":
        out = tmp.resample(resample_rule).min()
    else:
        out = tmp.resample(resample_rule).mean()
    # back to float seconds
    out.index = (out.index.view("i8") / 1e9).astype(float)
    out.index.name = "time_s"
    return out.dropna(how="all")


def decimate(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df.shape[0] <= max_points:
        return df
    stride = int(np.ceil(df.shape[0] / max_points))
    return df.iloc[::stride].copy()


def quick_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty or not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    return desc.rename(columns={"50%": "p50", "5%": "p05", "95%": "p95", "1%": "p01", "99%": "p99"})[
        ["count", "mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max"]
    ]


# ---------------------- UI ----------------------
st.title("Single-Run Time-Series Visualizer")

run_path = st.text_input("Run folder path", value="", placeholder="e.g. runs/20250909_120301__abcd1234")
load_clicked = st.button("Load run")  # optional, but keeps re-load explicit

if (not run_path) and (not load_clicked):
    st.info("Enter the path to a run folder and click **Load run**.")
    st.stop()

if load_clicked or run_path:
    df, meta, err = load_run(run_path)

    if err:
        st.error(err)
        st.stop()

    # Header info
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Metadata")
        st.json(meta)
    with c2:
        st.subheader("Summary")
        if df.empty:
            st.write("No data.")
        else:
            tmin, tmax = float(df.index.min()), float(df.index.max())
            st.metric("Signals", f"{df.shape[1]}")
            st.metric("Rows", f"{df.shape[0]:,}")
            st.write(f"**Time span:** {tmin:.6f}s → {tmax:.6f}s")

    if df.empty:
        st.stop()

    # Signal selection
    all_cols = list(df.columns)
    st.subheader("Signals")
    default_cols = all_cols#[: min(6, len(all_cols))]
    sel_cols = st.multiselect("Pick signals to plot", options=all_cols, default=default_cols)

    # Time & resampling controls
    tmin, tmax = float(df.index.min()), float(df.index.max())
    st.subheader("Time & Resampling")
    t0, t1 = st.slider(
        "Plot window (seconds)",
        min_value=tmin,
        max_value=tmax,
        value=(tmin, tmax),
        step=max((tmax - tmin) / 1000.0, 1e-6),
    )
    cols = st.columns(3)
    with cols[0]:
        resample_rule = st.text_input("Resample (optional)", value="", placeholder="e.g., 50ms, 200ms, 1s").strip() or None
    with cols[1]:
        agg = st.selectbox("Aggregation", ["mean", "median", "last", "first", "max", "min"], index=0)
    with cols[2]:
        max_points = st.number_input("Max points to plot (decimate)", min_value=1_000, max_value=200_000, value=15_000, step=1_000)

    # Prepare data
    cols_present = [c for c in sel_cols if c in df.columns] if sel_cols else all_cols
    view = df[cols_present]
    view = filter_and_resample(view, t0, t1, resample_rule, agg)
    view = decimate(view, int(max_points))

    # Plot
    st.subheader("Plot")
    if view.empty or not cols_present:
        st.info("No data to plot with the current filters.")
    else:
        fig = go.Figure()
        x = view.index.values
        for col in cols_present:
            fig.add_trace(go.Scatter(x=x, y=view[col].values, mode="lines", name=col))
        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Value",
            legend_title_text="",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Quick stats
    st.subheader("Quick stats")
    stats = quick_stats(view, cols_present)
    if stats.empty:
        st.info("No stats available (no data after filters).")
    else:
        st.dataframe(stats, use_container_width=True)
        st.caption(f"Rows displayed: {view.shape[0]}  |  Signals: {len(cols_present)}")

    # Export filtered view
    st.subheader("Export filtered view")
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = view.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="filtered.csv", mime="text/csv")
    with c2:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(view.reset_index())
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            parquet_bytes = sink.getvalue().to_pybytes()
            st.download_button(
                "⬇️ Download Parquet",
                data=parquet_bytes,
                file_name="filtered.parquet",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.warning(f"Parquet export unavailable ({e}).")