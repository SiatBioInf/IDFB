from pathlib import Path

import pandas as pd

from idfb.config import LUNG_SUBTYPES, TASK_OUTPUTS, TASK_SOURCES
from idfb.utils import (
    encode_platform,
    ensure_dir,
    load_header,
    read_csv_auto,
    read_expression_csv,
)


def load_lung_cancer_data(source_dir=None, subtypes=None):
    if source_dir is None:
        source_dir = TASK_SOURCES["Lung_cancer_subtybes"]
    source_dir = Path(source_dir)
    if subtypes is None:
        subtypes = list(LUNG_SUBTYPES)

    header = load_header()
    data_dir = source_dir / "data"
    subtype_info = read_csv_auto(
        source_dir / "stat_info.csv",
        usecols=[0, 1, 7],
    )
    platform_dict = dict(
        zip(subtype_info.iloc[:, 0].astype(str), subtype_info.iloc[:, 1].astype(str))
    )
    label_dict = dict(
        zip(subtype_info.iloc[:, 0].astype(str), subtype_info.iloc[:, -1].astype(str))
    )
    subtype_encoder = {s: i for i, s in enumerate(subtypes)}

    frames = []
    labels = []
    platforms = []

    files = sorted([f for f in data_dir.iterdir() if f.suffix.lower() == ".csv"])
    for path in files:
        print("-" * 20)
        print(path.name)
        gse_name = path.stem
        raw_label = label_dict.get(gse_name)
        if raw_label is None or str(raw_label) == "nan":
            print(f"skip (no subtype): {gse_name}")
            continue
        raw_label = str(raw_label).replace("非小细胞肺癌细胞", "非小细胞肺癌")
        if raw_label not in subtype_encoder:
            print(f"skip (unknown subtype={raw_label}): {gse_name}")
            continue

        try:
            expr = read_expression_csv(path, header)
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue

        platform = platform_dict.get(gse_name, "unknown")
        print(f"shape={expr.shape}, platform={platform}, subtype={raw_label}")
        frames.append(expr)
        labels.extend([subtype_encoder[raw_label]] * len(expr))
        platforms.extend([encode_platform(platform)] * len(expr))

    if not frames:
        raise RuntimeError("No lung cancer files processed")

    result = pd.concat(frames, axis=0, ignore_index=True)
    result = pd.concat(
        [
            result,
            pd.Series(platforms, name="GPL"),
            pd.Series(labels, name="Subtype"),
        ],
        axis=1,
    )
    print(f"Final shape: {result.shape}")
    print("Subtype encoder:", subtype_encoder)
    return result


def save_lung_cancer_processed():
    result = load_lung_cancer_data()
    out = ensure_dir(TASK_OUTPUTS["Lung_cancer_subtybes"]) / "processed_data.csv"
    result.to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    save_lung_cancer_processed()
