from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from idfb.config import HEADER_PATH, PLATFORM_ENCODER, REFERENCE_PLATFORM_ID


def load_header(path: Union[str, Path] = HEADER_PATH) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        return [line.strip() for line in f if line.strip()]


def encode_platform(name: str) -> int:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return len(PLATFORM_ENCODER)
    name = str(name).strip()
    if name in PLATFORM_ENCODER:
        return PLATFORM_ENCODER[name]
    return len(PLATFORM_ENCODER)


def encode_platform_for_model(name: str) -> int:
    code = encode_platform(name)
    if code >= len(PLATFORM_ENCODER):
        return REFERENCE_PLATFORM_ID
    return code


def resolve_platform(gse_name: str, platform_dict: dict) -> str:
    gse_name = str(gse_name)
    if "GPL" in gse_name:
        gpl = "GPL" + gse_name.split("GPL")[1]
        gpl = gpl.split("_")[0].split("-")[0]
        return gpl
    return platform_dict.get(gse_name, "unknown")


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_encoding(path: Union[str, Path]) -> str:
    path = Path(path)
    for enc in ["utf-8-sig", "utf-8", "gb2312", "gbk", "latin1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(8192)
            return enc
        except Exception:
            continue
    return "latin1"


def read_csv_auto(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    path = Path(path)
    enc = detect_encoding(path)
    return pd.read_csv(path, encoding=enc, **kwargs)


def _ensg_count(values) -> int:
    return sum(str(v).startswith("ENSG") for v in values)


def normalize_expression_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.map(lambda x: str(x).strip())
    df.columns = [str(c).strip() for c in df.columns]

    ensg_cols = _ensg_count(df.columns)
    ensg_idx = _ensg_count(df.index)

    if ensg_idx > ensg_cols:
        df = df.T

    gene_cols = [c for c in df.columns if str(c).startswith("ENSG")]
    if not gene_cols:
        raise ValueError("No ENSG columns found in expression matrix")

    df = df[gene_cols]
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T

    return df


def align_to_header(df: pd.DataFrame, header: List[str]) -> pd.DataFrame:
    df = df.reindex(columns=header)
    return df.fillna(df.median(numeric_only=True)).fillna(0.0).astype(np.float32)


def read_expression_csv(
    path: Union[str, Path],
    header: Optional[List[str]] = None,
) -> pd.DataFrame:
    if header is None:
        header = load_header()
    raw = read_csv_auto(path, index_col=0)
    expr = normalize_expression_matrix(raw)
    return align_to_header(expr, header)


def read_label_series(path: Union[str, Path], value_col: Optional[int] = -1) -> pd.Series:
    df = read_csv_auto(path, index_col=0)
    if df.shape[1] == 0:
        raise ValueError(f"No label column in {path}")
    series = df.iloc[:, value_col]
    series.index = series.index.map(lambda x: str(x).strip())
    return series


def align_labels(expr: pd.DataFrame, labels: pd.Series) -> pd.Series:
    labels = labels.copy()
    labels.index = labels.index.map(lambda x: str(x).strip())
    common = [i for i in expr.index if i in labels.index]
    if len(common) == 0:
        if len(labels) == len(expr):
            out = labels.copy()
            out.index = expr.index
            return out
        raise ValueError(
            f"Cannot align labels: expr={len(expr)}, labels={len(labels)}, overlap=0"
        )
    return labels.loc[common]


def gene_absmax_scale(x, eps: float = 1e-8):
    """Per-gene abs-max scale used by pseudo bulk (values roughly in [-1, 1])."""
    arr = np.asarray(x, dtype=np.float64)
    scale = np.maximum(np.max(np.abs(arr), axis=0, keepdims=True), eps)
    return (arr / scale).astype(np.float32), scale.astype(np.float32)


def set_seed(seed: int):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
