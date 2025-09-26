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


# Integrate with your folder layout
def main(index: int):
    available_scenarios = ["BouncingBall", "MassSpringDamperWithPIDController"]
    root_path = Path("models") / available_scenarios[index]
    # adjust name if different
    mdl_or_bundle = root_path / "simulink_model_original.mdl"

    parts = extract_simulink_parts_from_file(mdl_or_bundle)
    for key, item in parts.items():
        root = ET.fromstring(item)
        tree = ET.ElementTree(root)
        clean_tree(tree)
        pretty_indent(tree.getroot())
        # If you want the cleaned XML back as a string instead of saving to file:
        cleaned_xml = ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")
        
if __name__ == "__main__":
    main(index=1)