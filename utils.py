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

def sanity_check(uST, metadata_dataset=None, root_folder=Path(".")):
    assert metadata_dataset is not None 
    ### Efficiency 
    if uST is None:
        raise ValueError("uST and stop_time must be provided.")
    if uST <= 0:
        raise ValueError("uST must be > 0.")
    assert 0 < metadata_dataset["time_grid"]["fault_window"]["duration_fraction_range"][0] < metadata_dataset["time_grid"]["fault_window"]["duration_fraction_range"][1]
    assert metadata_dataset["time_grid"]["fault_window"]["duration_fraction_range"][1] < 1

    sim_file = root_folder / "sim_the_model.m"
    if not sim_file.exists():
        raise FileNotFoundError(f"Missing required file: {sim_file}")

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



from pathlib import Path
from typing import Dict, Iterable, Optional

WANTED = {
    "/simulink/systems/system_root.xml"
}

def iter_mwopc_parts(text: str):
    """
    Yields (path, content) for each part in a __MWOPC_PART_BEGIN__-delimited file.
    A part starts with a line: '__MWOPC_PART_BEGIN__ <path>' followed by its content
    until the next marker or EOF.
    """
    marker = "__MWOPC_PART_BEGIN__"
    # split while keeping nothing before the first marker
    chunks = text.split(marker)
    for chunk in chunks[1:]:
        # chunk begins with: ' <path>\\n<content...>'
        # grab the first non-empty line as the path
        chunk = chunk.lstrip()  # trim leading newlines/spaces
        if not chunk:
            continue
        first_line, _, rest = chunk.partition("\n")
        path = first_line.strip()
        content = rest.lstrip("\r\n")
        yield path, content

def extract_simulink_parts_from_file(file_path: Path,
                                     wanted_paths: Iterable[str] = WANTED,
                                     encoding: str = "utf-8",
                                     errors: str = "replace") -> Dict[str, str]:
    """
    Returns a dict: {wanted_path: xml_content}. Missing parts are simply omitted.
    """
    raw = file_path.read_text(encoding=encoding, errors=errors)
    out: Dict[str, str] = {}
    wanted_set = set(wanted_paths)
    for path, content in iter_mwopc_parts(raw):
        print(path)
        if path in wanted_set:
            out[path] = content
    return out

def write_parts(parts: Dict[str, str], out_dir: Path) -> None:
    """
    Writes each extracted part under out_dir mirroring its path.
    E.g. '/simulink/blockdiagram.xml' -> <out_dir>/simulink/blockdiagram.xml
    """
    for path, xml in parts.items():
        # normalize: remove leading slash to create a relative path
        rel = path[1:] if path.startswith("/") else path
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(xml, encoding="utf-8")


#!/usr/bin/env python3
"""
clean_simulink_xml.py

Strip Simulink/Simscape XML down to what matters for understanding system behavior:
- Keep: block types, key parameters (mass, spring, damper, PID gains, command, converter filter, etc.),
        solver parameters, and ALL connectivity (Src/Dst).
- Remove: UI/layout/appearance, logging flags, version/IDs/paths, scope cosmetics,
          line geometry, and PID datatype/range boilerplate.

Usage:
  python clean_simulink_xml.py --in model.xml --out model_clean.xml
"""

import argparse
import xml.etree.ElementTree as ET

# ---------- Config: what to remove ----------
TOP_LEVEL_P_REMOVE = {
    "Location", "Open", "ZoomFactor", "ReportName", "SIDHighWatermark",
}

# Shallow per-block cosmetic/ID props to drop (never affect behavior)
BLOCK_P_REMOVE = {
    "Position", "ZOrder", "BlockRotation", "BlockMirror", "BackgroundColor",
    "NameLocation", "ShowName", "HideAutomaticName", "FontSize",
    "LibraryVersion", "SchemaVersion", "ClassName",
    "ComponentPath", "ComponentVariants", "ComponentVariantNames",
    "SourceFile",
}

# InstanceData-level generic non-behavior fields (UI/codegen/logging)
INSTANCE_P_REMOVE = {
    "LogSimulationData",
    "SimscapeInstrumentationLogging",
    "SimscapeInstrumentationVariables",
    "InternalSimscapePortConfiguration",
    # UI/codegen boilerplate seen in many blocks:
    "RTWMemSecFuncInitTerm", "RTWMemSecFuncExecute",
    "RTWMemSecDataConstants", "RTWMemSecDataInternal",
    "RTWMemSecDataParameters",
    "ContentPreviewEnabled",
}

# Scope-only cosmetics to purge (entirely non-dynamic)
SCOPE_P_REMOVE = {
    "GraphicalSettings", "WindowPosition", "ScopeFrameLocation",
    "WasSavedAsWebScope", "Floating", "MultipleDisplayCache",
    "Title", "ActiveDisplayYMinimum", "ActiveDisplayYMaximum",
    "NumInputPorts",  # cosmetics for viewing
}

# PortProperties: keep names, drop logging flags
PORT_P_REMOVE = {"DataLogging"}

# Elements that are purely cosmetic/bookkeeping
ELEMENT_TAGS_REMOVE_UNDER_BLOCK = {"PortCounts"}

# In lines/branches, keep only Src/Dst/Name; drop geometry + cosmetics
LINE_BRANCH_KEEP_P = {"Src", "Dst", "Name"}
LINE_BRANCH_ATTR_REMOVE = {"ConnectType"}  # attribute on <Branch>

def is_pid_block(block: ET.Element) -> bool:
    for p in block.findall("P"):
        if p.get("Name") == "SourceBlock" and "slpidlib/PID Controller" in (p.text or ""):
            return True
    return False

def prune_pid_boilerplate(block: ET.Element) -> None:
    """
    Inside a PID Controller block, strip the massive datatype/range/variant boilerplate that
    doesn’t change behavior. Keep gains, limits, anti-windup selection, etc.
    """
    inst = block.find("InstanceData")
    if inst is None:
        return
    # Broad but *scoped* patterns so we don't touch non-PID blocks.
    to_delete = []
    for p in inst.findall("P"):
        name = p.get("Name", "")
        if (
            name.endswith("DataTypeStr")
            or name.endswith("OutMin") or name.endswith("OutMax")
            or name.endswith("ParamMin") or name.endswith("ParamMax")
            or name.endswith("AccumDataTypeStr")
            or name.endswith("ICVariant")
            or name.endswith("Variant")
            or name in {
                "RndMeth", "SaturateOnIntegerOverflow", "LockScale",
                "IntegratorRTWStateStorageClass", "FilterRTWStateStorageClass",
                "LinearizeAsGain",  # visualization/linearization hint
            }
        ):
            to_delete.append(p)

    for p in to_delete:
        inst.remove(p)

def remove_selected_Ps(elem: ET.Element, names: set) -> None:
    for p in list(elem.findall("P")):
        if p.get("Name") in names:
            elem.remove(p)

def clean_block(block: ET.Element) -> None:
    # Drop cosmetic/ID attributes
    if "SID" in block.attrib:
        del block.attrib["SID"]

    # Remove cosmetic subelements
    for tag in list(block):
        if tag.tag in ELEMENT_TAGS_REMOVE_UNDER_BLOCK:
            block.remove(tag)

    # Remove generic per-block P’s
    remove_selected_Ps(block, BLOCK_P_REMOVE)

    # Clean InstanceData
    inst = block.find("InstanceData")
    if inst is not None:
        remove_selected_Ps(inst, INSTANCE_P_REMOVE)

    # Prune PID-specific boilerplate
    if is_pid_block(block):
        prune_pid_boilerplate(block)

    # Special-case: Scope block cosmetics + <List> of IO strings
    name = block.get("Name", "")
    if block.tag == "Block" and name == "Scope":
        remove_selected_Ps(block, SCOPE_P_REMOVE)
        # Remove the List element that only configures scope inputs/labels
        for child in list(block):
            if child.tag == "List":
                block.remove(child)

    # PortProperties: remove logging flags only
    portprops = block.find("PortProperties")
    if portprops is not None:
        for port in portprops.findall("Port"):
            remove_selected_Ps(port, PORT_P_REMOVE)

def clean_line_or_branch(elem: ET.Element) -> None:
    # Remove unwanted attributes (e.g., ConnectType on <Branch>)
    for attr in LINE_BRANCH_ATTR_REMOVE:
        if attr in elem.attrib:
            del elem.attrib[attr]

    # Remove cosmetic P’s, keep only Src/Dst/Name
    for p in list(elem.findall("P")):
        if p.get("Name") not in LINE_BRANCH_KEEP_P:
            elem.remove(p)

    # Recurse into nested <Branch> elements
    for br in elem.findall("Branch"):
        clean_line_or_branch(br)

def pretty_indent(elem: ET.Element, level: int = 0) -> None:
    """
    ET.indent exists in Python 3.9+, but this keeps compatibility.
    """
    indent_str = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        for child in elem:
            pretty_indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str

def clean_tree(tree: ET.ElementTree) -> None:
    root = tree.getroot()

    # Clean top-level cosmetic P’s
    remove_selected_Ps(root, TOP_LEVEL_P_REMOVE)

    # Clean all Blocks
    for block in root.findall("Block"):
        clean_block(block)

    # Clean Lines and their Branches
    for line in root.findall("Line"):
        # Drop top-level cosmetic P’s on <Line> (ZOrder/Points/Labels):
        clean_line_or_branch(line)


def obtain_xml_system_description(mld_file_path: Path):
    parts = extract_simulink_parts_from_file(mld_file_path)
    #for key, item in parts.items():
    item = parts["/simulink/systems/system_root.xml"]
    root = ET.fromstring(item)
    tree = ET.ElementTree(root)
    clean_tree(tree)
    pretty_indent(tree.getroot())
    # If you want the cleaned XML back as a string instead of saving to file:
    cleaned_xml = ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")

    return cleaned_xml