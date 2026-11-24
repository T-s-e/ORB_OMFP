"""Portable array storage and provenance records."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA, RESULTS


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def processed_path(task):
    return DATA / "processed" / f"{task}.npz"


def data(task):
    with np.load(processed_path(task), allow_pickle=False) as archive:
        return dict(archive)


def save_table(name, rows, folder="tables"):
    dest = RESULTS / folder / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False, float_format="%.17g")


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(obj), indent=2, allow_nan=False) + "\n")


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("infinity" if value > 0 else "-infinity")
    return str(value) if isinstance(value, Path) else value
