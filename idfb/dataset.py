from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from idfb.config import (
    INPUT_DIM,
    PLATFORMS,
    PSEUDO_DIR,
    PSEUDO_NUM_SAMPLES,
    SEED,
    SPLIT_BY_DESIGN,
    TRAIN_TEST_RATIO,
)
from idfb.utils import encode_platform

PSEUDO_CACHE_DIR = PSEUDO_DIR / "_array_cache"
_CHUNK = 2000


def _platform_cache_paths(gpl: str) -> Tuple[Path, Path]:
    d = PSEUDO_CACHE_DIR
    return d / f"{gpl}_x.npy", d / f"{gpl}_design.npy"


def _csv_newer_than(csv_path: Path, *others: Path) -> bool:
    if not all(p.exists() for p in others):
        return True
    t = csv_path.stat().st_mtime
    return any(t > p.stat().st_mtime for p in others)


def cache_platform_arrays(gpl: str) -> Tuple[Path, Path]:
    """Convert mixed_samples.csv to float32 npy (one platform at a time)."""
    csv_path = PSEUDO_DIR / gpl / "mixed_samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing pseudo data: {csv_path}. Run pseudo generation first."
        )
    x_path, d_path = _platform_cache_paths(gpl)
    PSEUDO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not _csv_newer_than(csv_path, x_path, d_path):
        try:
            n_cached = int(np.load(x_path, mmap_mode="r").shape[0])
        except Exception:
            n_cached = -1
        if n_cached == int(PSEUDO_NUM_SAMPLES):
            return x_path, d_path
        print(f"  cache {gpl} has n={n_cached}, expected {PSEUDO_NUM_SAMPLES}; recache")

    print(f"Caching {gpl} mixed_samples.csv -> npy ...", flush=True)
    xs: List[np.ndarray] = []
    ds: List[np.ndarray] = []
    n = 0
    for chunk in pd.read_csv(csv_path, chunksize=_CHUNK, on_bad_lines="skip"):
        if "GPL" in chunk.columns:
            chunk = chunk.drop(columns=["GPL"])
        if "design_id" in chunk.columns:
            ds.append(chunk["design_id"].to_numpy(np.int64, copy=True))
            chunk = chunk.drop(columns=["design_id"])
        arr = chunk.to_numpy(np.float32, copy=True)
        if arr.shape[1] != INPUT_DIM:
            raise ValueError(
                f"{gpl}: gene dim {arr.shape[1]} != INPUT_DIM {INPUT_DIM}"
            )
        xs.append(arr)
        n += len(arr)
        if n % 10000 == 0:
            print(f"  {gpl}: {n} rows", flush=True)
        del chunk
    if not xs:
        raise ValueError(f"Empty pseudo file: {csv_path}")
    x = np.concatenate(xs, axis=0)
    del xs
    if ds:
        design = np.concatenate(ds, axis=0)
    else:
        design = np.arange(len(x), dtype=np.int64)
    np.save(x_path, x)
    np.save(d_path, design)
    print(f"  saved {x_path.name} shape={x.shape}", flush=True)
    del x, design, ds
    return x_path, d_path


def load_pseudo_arrays(
    gpls: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x (float32), platform codes, design_id. Uses npy cache."""
    if gpls is None:
        gpls = list(PLATFORMS)
    xs, ps, ds = [], [], []
    for gpl in gpls:
        x_path, d_path = cache_platform_arrays(gpl)
        x = np.load(x_path, mmap_mode=None)
        d = np.load(d_path, mmap_mode=None)
        p = np.full(len(x), encode_platform(gpl), dtype=np.int64)
        print(f"Loaded {gpl}: n={len(x)} genes={x.shape[1]}", flush=True)
        xs.append(x)
        ps.append(p)
        ds.append(d)
    x = np.concatenate(xs, axis=0)
    p = np.concatenate(ps, axis=0)
    design = np.concatenate(ds, axis=0)
    del xs, ps, ds
    print(f"Pseudo total: n={len(x)} platforms={len(gpls)}", flush=True)
    return x, p, design


def load_pseudo_data(
    gpls: Optional[List[str]] = None, keep_design_id: bool = False
) -> pd.DataFrame:
    x, p, design = load_pseudo_arrays(gpls)
    df = pd.DataFrame(x)
    df["GPL"] = p
    if keep_design_id:
        df["design_id"] = design
    return df


class DesignPairDataset(Dataset):
    """Each item is one sample plus a same-design partner from another platform."""

    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        design_ids: np.ndarray,
        seed: int = SEED,
    ):
        self.x = torch.tensor(x).float()
        self.p = torch.tensor(p).long()
        self.design_ids = torch.tensor(design_ids).long()
        self.rng = np.random.default_rng(seed)

        self.design_to_indices: Dict[int, List[int]] = {}
        for i, d in enumerate(design_ids.tolist()):
            self.design_to_indices.setdefault(int(d), []).append(i)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        d = int(self.design_ids[idx])
        candidates = [
            j for j in self.design_to_indices[d] if j != idx and int(self.p[j]) != int(self.p[idx])
        ]
        if not candidates:
            candidates = [j for j in self.design_to_indices[d] if j != idx]
        if not candidates:
            partner = idx
        else:
            partner = int(self.rng.choice(candidates))
        return (
            self.x[idx],
            self.p[idx],
            self.design_ids[idx],
            self.x[partner],
            self.p[partner],
        )


def _split_arrays(
    x: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    test_size: float,
    seed: int,
    split_by_design: bool,
):
    if split_by_design:
        if groups is None:
            raise ValueError("groups required when split_by_design=True")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(x, y, groups))
        print(
            f"Split by design_id: train_designs="
            f"{len(np.unique(groups[train_idx]))}, "
            f"val_designs={len(np.unique(groups[test_idx]))}"
        )
        return (
            x[train_idx],
            x[test_idx],
            y[train_idx],
            y[test_idx],
            groups[train_idx],
            groups[test_idx],
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=seed,
    )
    return x_train, x_test, y_train, y_test, None, None


def build_dataloaders(
    batch_size: int,
    gpls: Optional[List[str]] = None,
    test_size: float = TRAIN_TEST_RATIO,
    seed: int = SEED,
    split_by_design: bool = SPLIT_BY_DESIGN,
    with_design_partners: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    x, y, groups = load_pseudo_arrays(gpls)
    x_train, x_test, y_train, y_test, g_train, g_test = _split_arrays(
        x, y, groups, test_size, seed, split_by_design
    )
    del x, y, groups

    if with_design_partners and g_train is not None and g_test is not None:
        train_loader = DataLoader(
            DesignPairDataset(x_train, y_train, g_train, seed=seed),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            DesignPairDataset(x_test, y_test, g_test, seed=seed + 1),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train).float(),
                torch.tensor(y_train).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_test).float(),
                torch.tensor(y_test).long(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    return train_loader, test_loader


def load_task_tensors(csv_path: Path):
    df = pd.read_csv(csv_path)
    x = torch.tensor(df.iloc[:, :-2].values).float()
    p = torch.tensor(df.iloc[:, -2].values).long()
    y = torch.tensor(df.iloc[:, -1].values).long()
    return x, p, y
