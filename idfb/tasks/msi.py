from pathlib import Path

import numpy as np
import pandas as pd

from idfb.config import DATASET_DIR, TASK_OUTPUTS, TASK_SOURCES
from idfb.utils import (
    align_labels,
    encode_platform,
    ensure_dir,
    load_header,
    read_csv_auto,
    read_expression_csv,
    read_label_series,
    resolve_platform,
)


def _load_platform_dict(stat_path: Path) -> dict:
    stat = read_csv_auto(stat_path)
    sample_col = "Sample" if "Sample" in stat.columns else stat.columns[0]
    gpl_col = "GPL" if "GPL" in stat.columns else stat.columns[1]
    return dict(zip(stat[sample_col].astype(str), stat[gpl_col].astype(str)))


def load_msi_data(source_dir=None):
    if source_dir is None:
        source_dir = TASK_SOURCES["MSI"]
    source_dir = Path(source_dir)
    data_dir = source_dir / "data"
    header = load_header()
    platform_dict = _load_platform_dict(source_dir / "stat_info.csv")

    expr_files = sorted(
        [
            f
            for f in data_dir.iterdir()
            if f.name.startswith("GSE")
            and f.suffix.lower() == ".csv"
            and not f.name.endswith("-msi.csv")
        ]
    )

    frames = []
    labels = []
    platforms = []

    for path in expr_files:
        print("-" * 20)
        print(path.name)
        label_path = data_dir / path.name.replace(".csv", "-msi.csv")
        if not label_path.exists():
            print(f"skip (no label): {path.name}")
            continue

        expr = read_expression_csv(path, header)
        lab = read_label_series(label_path)
        lab = align_labels(expr, lab)
        expr = expr.loc[lab.index]

        lab = pd.to_numeric(lab, errors="coerce").fillna(0).astype(int)
        lab = (lab > 0).astype(int)

        gse_name = path.stem
        platform = resolve_platform(gse_name, platform_dict)
        print(f"shape={expr.shape}, platform={platform}, msi={lab.value_counts().to_dict()}")

        frames.append(expr)
        labels.extend(lab.tolist())
        platforms.extend([platform] * len(expr))

    if not frames:
        raise RuntimeError("No MSI files processed")

    result = pd.concat(frames, axis=0, ignore_index=True)
    result = pd.concat(
        [
            result,
            pd.Series([encode_platform(p) for p in platforms], name="GPL"),
            pd.Series(labels, name="MSI"),
        ],
        axis=1,
    )
    return result


def save_msi_processed():
    result = load_msi_data()
    out = ensure_dir(TASK_OUTPUTS["MSI"]) / "processed_data.csv"
    result.to_csv(out, index=False)
    print(f"Shape: {result.shape}")
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    save_msi_processed()
