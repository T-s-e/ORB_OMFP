"""Paper coordinates, file locations and frozen random draws."""

import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ORB_DATA_DIR", ROOT / "data"))
RESULTS = Path(os.environ.get("ORB_RESULTS_DIR", ROOT / "results"))
REFERENCE = ROOT / "references"
MEASUREMENTS = ROOT / "measurements"
PROTOCOL = json.loads((REFERENCE / "protocol.json").read_text())
GROUPS = {"GW": ["gw", "gw_bond"], "FRED": ["fred_equity"], "Treasury": ["macro_treasury"]}
TASKS = tuple(t for ts in GROUPS.values() for t in ts)
Q = (0.90, 0.95, 0.99)
U = np.r_[0.0, np.geomspace(1e-7, 1e8, 601), np.inf]
DRAWS = 128


def anchor(task):
    return PROTOCOL["kernel"][task]
