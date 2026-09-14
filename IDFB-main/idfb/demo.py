from pathlib import Path

import numpy as np
import pandas as pd

from idfb.config import CELL_TYPES, DATASET_DIR, MODEL_DIR, PLATFORMS, PSEUDO_DIR
from idfb.evaluate import run_evaluate
from idfb.integrate import run_integrate
from idfb.train import run_train
from idfb.utils import ensure_dir


def create_demo_raw_data(
    n_genes=256,
    n_pseudo_per_platform=48,
    n_task_samples=72,
    seed=42,
):
    rng = np.random.default_rng(seed)
    gene_cols = [f"G{i}" for i in range(n_genes)]

    for p_idx, gpl in enumerate(PLATFORMS):
        platform_dir = ensure_dir(PSEUDO_DIR / gpl)
        for c_idx, cell_type in enumerate(CELL_TYPES):
            cell_dir = ensure_dir(platform_dir / cell_type)
            values = rng.random((10, n_genes)) * (0.4 + 0.1 * c_idx)
            df = pd.DataFrame(values, columns=gene_cols)
            df.index = [f"{cell_type}_{i}" for i in range(10)]
            df.to_csv(cell_dir / "samples.csv")

        mixed = []
        for _ in range(n_pseudo_per_platform):
            sample = rng.random(n_genes) + 0.05 * p_idx
            mixed.append(np.clip(sample, 0, 1))
        out = platform_dir / "mixed_samples.csv"
        pd.DataFrame(np.vstack(mixed)).to_csv(out, index=False)
        print(f"Demo pseudo data: {out}")

    task_dir = ensure_dir(DATASET_DIR / "Demo")
    rows, platforms, labels = [], [], []
    for i in range(n_task_samples):
        p = i % len(PLATFORMS)
        label = i % 2
        expr = rng.random(n_genes) + 0.08 * p + 0.12 * label
        rows.append(np.clip(expr, 0, None))
        platforms.append(p)
        labels.append(label)

    df = pd.DataFrame(np.vstack(rows), columns=gene_cols)
    df["GPL"] = platforms
    df["Label"] = labels
    out = task_dir / "processed_data.csv"
    df.to_csv(out, index=False)
    print(f"Demo task data: {out}")
    return n_genes


def run_demo(epochs=5, batch_size=16, latent_dim=64, n_genes=256):
    print("=== IDFB Demo ===")
    create_demo_raw_data(n_genes=n_genes)
    print("\n--- Training ---")
    run_train(
        batch_size=batch_size,
        n_epochs=epochs,
        latent_dim=latent_dim,
        input_dim=n_genes,
        early_stopping=False,
    )
    print("\n--- Integrate ---")
    run_integrate("Demo")
    print("\n--- Evaluate ---")
    run_evaluate("Demo")
    print(f"\nModels: {MODEL_DIR}")
    print("Demo finished.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run IDFB end-to-end demo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-genes", type=int, default=256)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.data_only:
        create_demo_raw_data(n_genes=args.n_genes)
    else:
        run_demo(
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            n_genes=args.n_genes,
        )


if __name__ == "__main__":
    main()
