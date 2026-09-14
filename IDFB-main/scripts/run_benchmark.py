"""Compare IDFB vs bulk baselines on BC/PM/OIS and DEG overlap."""
import argparse
from pathlib import Path

import torch  # noqa: F401  Windows: import torch before numpy
import numpy as np
import pandas as pd

from idfb.baselines import apply_method
from idfb.config import DATASET_DIR, MODEL_DIR
from idfb.deg import deg_report
from idfb.evaluate import aggregate_scores, compute_raw_metrics
from idfb.utils import ensure_dir

DEFAULT_METHODS = [
    "Uncorrected",
    "limma",
    "ComBat",
    "SVA",
    "RUVg",
    "Harmony",
    "scVI",
    "ResPAN",
    "IDFB",
]


def load_task(task: str):
    path = DATASET_DIR / task / "processed_data.csv"
    df = pd.read_csv(path)
    x = df.iloc[:, :-2].to_numpy(dtype=np.float64)
    p = df.iloc[:, -2]
    y = df.iloc[:, -1]
    return x, p, y, df.columns[:-2]


def load_idfb(task: str, n_genes: int) -> np.ndarray:
    path = DATASET_DIR / task / "generated_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run integrate first.")
    arr = pd.read_csv(path).to_numpy(dtype=np.float64)
    if arr.shape[1] != n_genes:
        raise ValueError(f"IDFB matrix genes {arr.shape[1]} != {n_genes}")
    return arr


def run_task(task: str, methods, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 60)
    print("TASK", task)
    X, p, y, gene_cols = load_task(task)
    print(f"n={X.shape[0]} genes={X.shape[1]} platforms={p.nunique()} labels={y.nunique()}")

    matrices = {}
    for name in methods:
        print(f"  correcting: {name} ...", flush=True)
        if name.lower() == "idfb":
            matrices[name] = load_idfb(task, X.shape[1])
            continue
        try:
            matrices[name] = apply_method(name, X, p, y)
        except ImportError as e:
            print(f"  skip {name}: {e}")
            continue
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}")
            continue

    raw = {}
    for name, Xm in matrices.items():
        print(f"  scoring: {name} ...", flush=True)
        raw[name] = compute_raw_metrics(X, Xm, y, p)

    from idfb.evaluate import score_from_raw

    scored = aggregate_scores(raw, apply_minmax=True)
    rows = []
    for name in scored:
        r = scored[name]
        raw_agg = score_from_raw(raw[name])
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
                "BC_raw": raw_agg["bc"],
                "PM_raw": raw_agg["pm"],
                "OIS_raw": raw_agg["ois"],
                "BC": r["bc"],
                "PM": r["pm"],
                "OIS": r["ois"],
            }
        )
    metrics = pd.DataFrame(rows).sort_values("OIS", ascending=False)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("  DEG overlap ...", flush=True)
    deg = deg_report(X, y, p, matrices)
    deg.insert(0, "task", task)
    print(deg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ensure_dir(out_dir)
    metrics.to_csv(out_dir / f"{task}_metrics.csv", index=False)
    deg.to_csv(out_dir / f"{task}_deg.csv", index=False)
    return metrics, deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["MSI", "Lung_cancer_subtybes", "Cancertype", "Survival_analysis"],
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--out",
        type=str,
        default=str(MODEL_DIR / "benchmark"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    all_m, all_d = [], []
    for task in args.tasks:
        m, d = run_task(task, args.methods, out_dir)
        all_m.append(m)
        all_d.append(d)
    metrics = pd.concat(all_m, ignore_index=True)
    degs = pd.concat(all_d, ignore_index=True)
    metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    degs.to_csv(out_dir / "all_deg.csv", index=False)
    print("=" * 60)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
