from pathlib import Path

import pandas as pd

from idfb.config import CANCER_TYPES, TASK_OUTPUTS, TASK_SOURCES
from idfb.utils import (
    encode_platform,
    ensure_dir,
    load_header,
    read_csv_auto,
    read_expression_csv,
    resolve_platform,
)


def load_cancer_type_data(source_dir=None, cancers=None):
    if source_dir is None:
        source_dir = TASK_SOURCES["Cancertype"]
    source_dir = Path(source_dir)
    if cancers is None:
        cancers = list(CANCER_TYPES)

    header = load_header()
    stat = read_csv_auto(source_dir / "stat_info.csv")
    platform_dict = dict(zip(stat["Sample"].astype(str), stat["GPL"].astype(str)))
    cancer_encoder = {c: i for i, c in enumerate(cancers)}

    frames = []
    labels = []
    platforms = []

    for cancer_type in cancers:
        folder = source_dir / cancer_type
        if not folder.exists():
            print(f"missing cancer folder: {folder}")
            continue
        print(f"\nCancer type: {cancer_type}")
        files = sorted(
            [
                f
                for f in folder.iterdir()
                if f.name.startswith("GSE") and f.suffix.lower() == ".csv"
            ]
        )
        for path in files:
            print("-" * 20)
            print(path.name)
            try:
                expr = read_expression_csv(path, header)
            except Exception as e:
                print(f"skip {path.name}: {e}")
                continue
            gse_name = path.stem
            platform = resolve_platform(gse_name, platform_dict)
            print(f"shape={expr.shape}, platform={platform}")
            frames.append(expr)
            labels.extend([cancer_encoder[cancer_type]] * len(expr))
            platforms.extend([encode_platform(platform)] * len(expr))

    if not frames:
        raise RuntimeError("No cancertype files processed")

    result = pd.concat(frames, axis=0, ignore_index=True)
    result = pd.concat(
        [
            result,
            pd.Series(platforms, name="GPL"),
            pd.Series(labels, name="CancerType"),
        ],
        axis=1,
    )
    print(f"\nFinal shape: {result.shape}")
    print("Cancer encoder:", cancer_encoder)
    return result


def save_cancertype_processed():
    result = load_cancer_type_data()
    out = ensure_dir(TASK_OUTPUTS["Cancertype"]) / "processed_data.csv"
    result.to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    save_cancertype_processed()
