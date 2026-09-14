from pathlib import Path

import pandas as pd

from idfb.config import TASK_OUTPUTS, TASK_SOURCES
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


def load_survival_data(source_dir=None):
    if source_dir is None:
        source_dir = TASK_SOURCES["Survival_analysis"]
    source_dir = Path(source_dir)
    data_dir = source_dir / "data"
    header = load_header()
    stat = read_csv_auto(source_dir / "stat_info.csv")
    platform_dict = dict(zip(stat["Sample"].astype(str), stat["GPL"].astype(str)))

    expr_files = sorted(
        [
            f
            for f in data_dir.iterdir()
            if f.name.startswith("GSE")
            and f.suffix.lower() == ".csv"
            and not f.name.endswith("_os.csv")
        ]
    )

    frames = []
    labels = []
    platforms = []

    for path in expr_files:
        print("-" * 20)
        print(path.name)
        os_path = data_dir / path.name.replace(".csv", "_os.csv")
        if not os_path.exists():
            print(f"skip (no os): {path.name}")
            continue

        try:
            expr = read_expression_csv(path, header)
            lab = read_label_series(os_path)
            lab = align_labels(expr, lab)
            expr = expr.loc[lab.index]
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue

        lab = pd.to_numeric(lab, errors="coerce").fillna(0).astype(int)
        gse_name = path.stem
        platform = resolve_platform(gse_name, platform_dict)
        print(f"shape={expr.shape}, platform={platform}, os={lab.value_counts().to_dict()}")

        frames.append(expr)
        labels.extend(lab.tolist())
        platforms.extend([encode_platform(platform)] * len(expr))

    if not frames:
        raise RuntimeError("No survival files processed")

    result = pd.concat(frames, axis=0, ignore_index=True)
    result = pd.concat(
        [
            result,
            pd.Series(platforms, name="GPL"),
            pd.Series(labels, name="Survival"),
        ],
        axis=1,
    )
    print(f"Final shape: {result.shape}")
    return result


def save_survival_processed():
    result = load_survival_data()
    out = ensure_dir(TASK_OUTPUTS["Survival_analysis"]) / "processed_data.csv"
    result.to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    save_survival_processed()
