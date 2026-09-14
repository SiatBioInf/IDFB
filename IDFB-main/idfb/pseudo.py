from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from idfb.config import (
    CELL_TYPES,
    PLATFORMS,
    PSEUDO_DIR,
    PSEUDO_NUM_SAMPLES,
    PSEUDO_PLATFORM_SOURCES,
    SEED,
    USE_SAMPLE_MINMAX,
)
from idfb.utils import ensure_dir, load_header, read_expression_csv


def sample_ratio_vector(rng):
    cell_ratio = float(rng.uniform(0.2, 0.5))
    remaining = 1.0 - cell_ratio
    weights = rng.random(3)
    weights = weights / weights.sum() * remaining
    return np.array([cell_ratio, weights[0], weights[1], weights[2]], dtype=np.float64)


def load_cell_type_tables(source_folder: Path, header, exclude_substrings=None):
    """Load per-cell-type expression tables.

    exclude_substrings: optional list of filename substrings to drop (held-out
    cell-line files for generalization checks). Matching is case-insensitive.
    """
    exclude_substrings = [s.lower() for s in (exclude_substrings or [])]
    cells_data = {}
    for cell_type in CELL_TYPES:
        cell_dir = source_folder / cell_type
        if not cell_dir.exists():
            raise FileNotFoundError(f"Missing cell folder: {cell_dir}")
        frames = []
        skipped = 0
        for path in sorted(cell_dir.glob("*.csv")):
            if path.name.startswith("_"):
                continue
            name_l = path.name.lower()
            if any(ex in name_l for ex in exclude_substrings):
                skipped += 1
                continue
            try:
                expr = read_expression_csv(path, header)
                frames.append(expr)
            except Exception as e:
                print(f"skip {path}: {e}")
        if not frames:
            raise FileNotFoundError(
                f"No usable CSV in {cell_dir} after exclude={exclude_substrings}"
            )
        cells = pd.concat(frames, axis=0, ignore_index=True).fillna(0.0)
        cells_data[cell_type] = cells
        msg = f"  {cell_type}: {cells.shape}"
        if skipped:
            msg += f" (held-out files skipped={skipped})"
        print(msg)
    return cells_data


def write_mixed_csv(path, arr: np.ndarray, header, chunk: int = 500):
    """Write design_id + genes. Avoid pandas append (can glue rows on Windows)."""
    n = arr.shape[0]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("design_id," + ",".join(header) + "\n")
        ids = np.arange(n, dtype=np.float64).reshape(-1, 1)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = np.concatenate(
                [ids[start:end], arr[start:end].astype(np.float64, copy=False)],
                axis=1,
            )
            np.savetxt(f, block, delimiter=",", fmt="%.6g", newline="\n")
            print(f"  wrote {end}/{n} -> {path.name}", flush=True)


def mix_one_sample(cells_data, ratios, rng):
    combined = None
    for cell_type, ratio in zip(CELL_TYPES, ratios):
        idx = int(rng.integers(0, len(cells_data[cell_type])))
        part = cells_data[cell_type].iloc[idx].values * ratio
        combined = part if combined is None else combined + part
    return combined


def run_generate_pseudo(
    num_samples=PSEUDO_NUM_SAMPLES,
    seed=SEED,
    platforms=None,
    exclude_substrings=None,
    out_subdir=None,
    write_chunk=500,
):
    if platforms is None:
        platforms = list(PLATFORMS)
    header = load_header()
    rng = np.random.default_rng(seed)

    designs = np.vstack([sample_ratio_vector(rng) for _ in range(num_samples)])
    print(f"Shared designs: {num_samples}")
    print(f"Platforms: {platforms}")
    if exclude_substrings:
        print(f"Held-out filename filters: {exclude_substrings}")

    root_out = PSEUDO_DIR if out_subdir is None else PSEUDO_DIR / out_subdir
    ensure_dir(root_out)
    design_path = root_out / "shared_design_ratios.csv"
    pd.DataFrame(designs, columns=CELL_TYPES).to_csv(design_path, index_label="design_id")
    print(f"Saved shared ratios: {design_path}")

    # One platform at a time to bound memory (~3.5GB float32 for 45k×20k).
    for pi, gpl in enumerate(platforms):
        source = PSEUDO_PLATFORM_SOURCES.get(gpl)
        if source is None or not source.exists():
            raise FileNotFoundError(f"Missing pseudo source for {gpl}: {source}")
        print(f"\n=== Load source {gpl} ===", flush=True)
        print(f"source: {source}", flush=True)
        cells = load_cell_type_tables(source, header, exclude_substrings=exclude_substrings)
        # convert cell tables to float32 numpy for faster mix
        cell_arrs = {
            ct: cells[ct].to_numpy(dtype=np.float32, copy=False) for ct in CELL_TYPES
        }
        del cells

        pi = list(PLATFORMS).index(gpl) if gpl in PLATFORMS else pi
        local_rng = np.random.default_rng(seed + 17 + pi)
        arr = np.empty((num_samples, len(header)), dtype=np.float32)
        for i in range(num_samples):
            ratios = designs[i]
            combined = None
            for ct, ratio in zip(CELL_TYPES, ratios):
                idx = int(local_rng.integers(0, cell_arrs[ct].shape[0]))
                part = cell_arrs[ct][idx] * np.float32(ratio)
                combined = part if combined is None else combined + part
            arr[i] = combined
            if (i + 1) % 5000 == 0:
                print(f"  mixed {i+1}/{num_samples}", flush=True)

        del cell_arrs
        if USE_SAMPLE_MINMAX:
            arr = MinMaxScaler().fit_transform(arr.T).T.astype(np.float32)
        else:
            col_max = np.max(np.abs(arr), axis=0, keepdims=True)
            arr = arr / np.maximum(col_max, 1e-8)

        out_dir = ensure_dir(root_out / gpl)
        out = out_dir / "mixed_samples.csv"
        write_mixed_csv(out, arr, header, chunk=write_chunk)
        print(f"Saved: {out} shape={arr.shape}", flush=True)
        del arr

    print(f"USE_SAMPLE_MINMAX={USE_SAMPLE_MINMAX}")
    return root_out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=PSEUDO_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--platforms",
        type=str,
        nargs="*",
        default=None,
        help="Subset of GPL ids; default all",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Filename substrings to hold out (e.g. HCT116 DLD1)",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Optional subfolder under Pseudo data for held-out runs",
    )
    args = parser.parse_args()
    run_generate_pseudo(
        num_samples=args.num_samples,
        seed=args.seed,
        platforms=args.platforms,
        exclude_substrings=args.exclude,
        out_subdir=args.out_subdir,
    )


if __name__ == "__main__":
    main()
