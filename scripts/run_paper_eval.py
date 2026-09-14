"""Fig. 3-style evaluation: manuscript metrics + min–max BC/PM/OIS.

Methods in the paper: Limma, ComBat, Harmony, BERMAD, ResPAN, IDFB.
BERMAD / ResPAN are PCA-space adaptations of those models (official packages
target scRNA-seq AnnData). IDFB is not blended with SVA.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch  # noqa: F401  Windows: import torch before numpy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idfb.baselines import apply_method
from idfb.config import DATASET_DIR
from idfb.evaluate import aggregate_scores, compute_raw_metrics
from idfb.utils import ensure_dir

PAPER_METHODS = ["Limma", "ComBat", "Harmony", "BERMAD", "ResPAN", "IDFB"]
PAPER_TASKS = ["Cancertype", "Lung_cancer_subtybes", "MSI", "Survival_analysis"]


def load_task(task: str):
    path = DATASET_DIR / task / "processed_data.csv"
    df = pd.read_csv(path)
    x = df.iloc[:, :-2].to_numpy(dtype=np.float64)
    p = df.iloc[:, -2]
    y = df.iloc[:, -1]
    return x, p, y


def load_idfb(task: str, n_genes: int) -> np.ndarray:
    path = DATASET_DIR / task / "generated_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run integrate first.")
    arr = pd.read_csv(path).to_numpy(dtype=np.float64)
    if arr.shape[1] != n_genes:
        raise ValueError(f"IDFB matrix genes {arr.shape[1]} != {n_genes}")
    return arr


def run_task(task: str, methods, out_dir: Path) -> pd.DataFrame:
    print("=" * 60, flush=True)
    print("TASK", task, flush=True)
    X, p, y = load_task(task)
    print(
        f"n={X.shape[0]} genes={X.shape[1]} platforms={p.nunique()} labels={y.nunique()}",
        flush=True,
    )

    matrices = {}
    for name in methods:
        print(f"  correcting: {name} ...", flush=True)
        if name.lower() == "idfb":
            matrices[name] = load_idfb(task, X.shape[1])
            continue
        matrices[name] = apply_method(name, X, p, y)

    raw = {}
    for name, Xm in matrices.items():
        print(f"  scoring (paper metrics): {name} ...", flush=True)
        raw[name] = compute_raw_metrics(X, Xm, y, p, paper=True)

    scored = aggregate_scores(raw, apply_minmax=True)
    rows = []
    for name in scored:
        r = scored[name]
        rows.append(
            {
                "task": task,
                "method": name,
                "MAP": r["map"],
                "Cancer_ASW": r["cancer_asw"],
                "NC": r["nc"],
                "SAS": r["sas"],
                "GPL_ASW": r["gpl_asw"],
                "GC": r["gc"],
                "MAP_norm": r["map_norm"],
                "Cancer_ASW_norm": r["cancer_asw_norm"],
                "NC_norm": r["nc_norm"],
                "SAS_norm": r["sas_norm"],
                "GPL_ASW_norm": r["gpl_asw_norm"],
                "GC_norm": r["gc_norm"],
                "BC": r["bc"],
                "PM": r["pm"],
                "OIS": r["ois"],
            }
        )
    metrics = pd.DataFrame(rows).sort_values("OIS", ascending=False)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    ensure_dir(out_dir)
    metrics.to_csv(out_dir / f"{task}_metrics.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=PAPER_TASKS)
    parser.add_argument("--methods", nargs="+", default=PAPER_METHODS)
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "paper_eval_full"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    all_m = []
    for task in args.tasks:
        all_m.append(run_task(task, args.methods, out_dir))
    metrics = pd.concat(all_m, ignore_index=True)
    metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    summary = (
        metrics[["task", "method", "BC", "PM", "OIS"]]
        .sort_values(["task", "OIS"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "fig3_bc_pm_ois.csv", index=False)
    print("=" * 60, flush=True)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
