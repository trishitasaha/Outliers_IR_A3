from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUTPUT = Path(__file__).resolve().parent / "output"

def load_split(name: str, output_dir: Path = OUTPUT):
    d = np.load(Path(output_dir) / f"{name}.npz")
    info = {k: d[k] for k in ("grade", "query_ids", "doc_ids")}
    return d["X"], d["y"], info

class Scaler:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean), np.asarray(std)
    def transform(self, X):
        return (X - self.mean) / self.std

def load_scaler(output_dir: Path = OUTPUT) -> Scaler:
    s = json.loads((Path(output_dir) / "scaler.json").read_text())
    return Scaler(s["mean"], s["std"])
