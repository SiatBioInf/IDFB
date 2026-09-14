"""Integration metrics aligned with the manuscript.

Biology Conservation (BC):
  MAP, Cancer-type ASW, Neighborhood Consistency (NC)
  → min–max across methods (per metric), then mean.
  NC is the overlap of per-class kNN graphs before vs after integration.
  Uncorrected NC is 1 by construction.

Platform Mixing (PM):
  Seurat Alignment Score (SAS), GPL ASW, Graph Connectivity (GC)
  → min–max across methods (per metric), then mean.
  GC is the largest connected-component fraction within each biological class.

Overall Integration Score (OIS) = 0.6 * BC + 0.4 * PM

For a single method, BC/PM/OIS use the raw metric means (no min–max).
When comparing ≥2 methods, call aggregate_scores(...) to min–max each
metric across methods, then average and form OIS.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from idfb.config import DATASET_DIR

BIO_KEYS = ("map", "cancer_asw", "nc")
MIX_KEYS = ("sas", "gpl_asw", "gc")
OIS_W_BIO = 0.6
OIS_W_MIX = 0.4


def _as_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(np.asarray(y))


def _as_frame(x) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(x, pd.DataFrame):
        return x.reset_index(drop=True)
    return np.asarray(x)


def mean_average_precision(x, y, neighbor_frac=0.01) -> float:
    """MAP: class separability in neighbor space (higher = better biology)."""
    x = _as_frame(x)
    y = _as_series(y)
    n_samples = len(y)
    k = max(int(n_samples * neighbor_frac), 1)
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(x)

    class_aps = []
    for label in y.unique():
        class_samples = x[y.values == label]
        if len(class_samples) == 0:
            continue
        neighbors = knn.kneighbors(class_samples, return_distance=False)[:, 1:]
        neighbor_labels = y.values[neighbors]
        matches = neighbor_labels == label
        ap = np.mean([np.sum(match) / k for match in matches])
        class_aps.append(ap)
    return float(np.mean(class_aps)) if class_aps else 0.0


def cancer_type_asw(x, y) -> float:
    """Cancer-type ASW scaled to [0, 1]: (silhouette + 1) / 2."""
    y = _as_series(y)
    if y.nunique() < 2 or len(y) < 3:
        return 0.5
    sil = silhouette_score(_as_frame(x), y)
    return float((sil + 1.0) / 2.0)


def neighborhood_consistency(x_single, x_integrated, y, neighbor_frac=0.01) -> float:
    """NC: overlap of per-class neighbor graphs before vs after integration."""
    x_single = _as_frame(x_single)
    x_integrated = _as_frame(x_integrated)
    y = _as_series(y)
    k = max(int(len(y) * neighbor_frac), 1)

    nc_per_class = []
    for label in y.unique():
        mask = (y == label).values
        x_s = x_single[mask] if isinstance(x_single, pd.DataFrame) else x_single[mask]
        x_i = (
            x_integrated[mask]
            if isinstance(x_integrated, pd.DataFrame)
            else x_integrated[mask]
        )
        if len(x_s) <= k:
            continue
        nn_s = NearestNeighbors(n_neighbors=k + 1).fit(x_s).kneighbors_graph(x_s)
        nn_i = NearestNeighbors(n_neighbors=k + 1).fit(x_i).kneighbors_graph(x_i)
        nn_s.setdiag(0)
        nn_i.setdiag(0)
        intersection = nn_s.multiply(nn_i).sum(axis=1).A1
        union = (nn_s + nn_i).astype(bool).sum(axis=1).A1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(union > 0, intersection / union, 0.0)
        nc_per_class.append(float(np.mean(ratio)))
    return float(np.mean(nc_per_class)) if nc_per_class else 0.0


def neighborhood_consistency_paper(x_single, x_integrated, neighbor_frac=0.01) -> float:
    """NC as in the manuscript: Jaccard of kNN sets before vs after (k = 1% of n)."""
    x_s = np.asarray(_as_frame(x_single))
    x_i = np.asarray(_as_frame(x_integrated))
    n = int(x_s.shape[0])
    k = max(int(n * neighbor_frac), 1)
    if n <= k:
        return 1.0
    ind_s = NearestNeighbors(n_neighbors=k + 1).fit(x_s).kneighbors(x_s, return_distance=False)[:, 1:]
    ind_i = NearestNeighbors(n_neighbors=k + 1).fit(x_i).kneighbors(x_i, return_distance=False)[:, 1:]
    scores = []
    for a, b in zip(ind_s, ind_i):
        sa, sb = set(a.tolist()), set(b.tolist())
        union = len(sa | sb)
        scores.append((len(sa & sb) / union) if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def seurat_alignment_score(
    x,
    p,
    neighbor_frac=0.01,
    subsample_platforms: bool = False,
    seed: int = 0,
) -> float:
    """SAS: platform mixing in neighbors (higher = better mixing).

    Manuscript: downsample each GPL to the smallest platform size, then
    k = 1% of the subsampled n.
    """
    x = _as_frame(x)
    x_arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
    p = _as_series(p)
    n_platforms = int(p.nunique())
    if n_platforms < 2:
        return 1.0
    if subsample_platforms:
        counts = p.value_counts()
        n_min = int(counts.min())
        rng = np.random.RandomState(int(seed))
        parts = []
        for plat in p.unique():
            idx = np.where(p.values == plat)[0]
            parts.append(rng.choice(idx, size=n_min, replace=False))
        take = np.concatenate(parts)
        x_arr = x_arr[take]
        p = p.iloc[take].reset_index(drop=True)
    n_samples = len(p)
    k = max(int(n_samples * neighbor_frac), 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(x_arr)
    neighbors = nn.kneighbors(x_arr, return_distance=False)[:, 1:]
    neighbor_platforms = p.values[neighbors]
    same_platform_ratio = (neighbor_platforms == p.values[:, np.newaxis]).mean(axis=1)
    expected_ratio = 1.0 / n_platforms
    sas = 1.0 - (same_platform_ratio.mean() - expected_ratio) / (1.0 - expected_ratio)
    return float(max(0.0, min(1.0, sas)))


def gpl_asw(x, p, y=None) -> float:
    """GPL ASW: platform silhouette mapped so higher = less platform clustering.

    Manuscript: average the mixing score within each biological class.
    """
    x = _as_frame(x)
    p = _as_series(p)
    if y is None:
        if p.nunique() < 2 or len(p) < 3:
            return 1.0
        sil = silhouette_score(x, p)
        return float((1.0 - sil) / 2.0)
    y = _as_series(y)
    x_arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
    scores = []
    for label in y.unique():
        mask = (y == label).values
        if int(mask.sum()) < 3:
            continue
        p_sub = p[mask]
        if int(p_sub.nunique()) < 2:
            continue
        sil = silhouette_score(x_arr[mask], p_sub)
        scores.append((1.0 - sil) / 2.0)
    if not scores:
        return gpl_asw(x, p, y=None)
    return float(np.mean(scores))


def graph_connectivity(x, y, n_neighbors=15) -> float:
    """GC within each biological class: fraction in largest connected component."""
    x = _as_frame(x)
    y = _as_series(y)
    n_neighbors = min(n_neighbors, max(len(y) - 1, 1))
    if n_neighbors < 1:
        return 0.0

    adj = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors_graph(x)
    gc_scores = []
    for label in y.unique():
        mask = (y == label).values
        n_lab = int(mask.sum())
        if n_lab <= 1:
            continue
        sub = adj[mask][:, mask]
        n_components, comp_labels = connected_components(sub, directed=False)
        if n_components <= 0:
            continue
        largest_cc = int(np.max(np.bincount(comp_labels)))
        gc_scores.append(largest_cc / n_lab)
    return float(np.mean(gc_scores)) if gc_scores else 0.0


def compute_raw_metrics(
    x_single,
    x_integrated,
    y,
    p,
    neighbor_frac=0.01,
    n_neighbors=15,
    paper: bool = False,
) -> Dict[str, float]:
    """Return the six raw metrics (each already on an approximate [0, 1] scale)."""
    y = _as_series(y)
    p = _as_series(p)
    if paper:
        nc = neighborhood_consistency_paper(x_single, x_integrated, neighbor_frac)
        sas = seurat_alignment_score(
            x_integrated, p, neighbor_frac, subsample_platforms=True
        )
        gpl = gpl_asw(x_integrated, p, y=y)
    else:
        nc = neighborhood_consistency(x_single, x_integrated, y, neighbor_frac)
        sas = seurat_alignment_score(x_integrated, p, neighbor_frac)
        gpl = gpl_asw(x_integrated, p)
    return {
        "map": mean_average_precision(x_integrated, y, neighbor_frac),
        "cancer_asw": cancer_type_asw(x_integrated, y),
        "nc": nc,
        "sas": sas,
        "gpl_asw": gpl,
        "gc": graph_connectivity(x_integrated, y, n_neighbors),
    }


def minmax_normalize(values: Sequence[float]) -> List[float]:
    """Min–max across methods for one metric. Constant column → all 1.0."""
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return [1.0] * len(arr)
    return [float((v - lo) / (hi - lo)) for v in arr]


def score_from_raw(raw: Mapping[str, float]) -> Dict[str, float]:
    """BC / PM / OIS from raw metrics (no cross-method min–max)."""
    bc = float(np.mean([float(raw[k]) for k in BIO_KEYS]))
    pm = float(np.mean([float(raw[k]) for k in MIX_KEYS]))
    ois = OIS_W_BIO * bc + OIS_W_MIX * pm
    return {
        **{k: float(raw[k]) for k in list(BIO_KEYS) + list(MIX_KEYS)},
        "bc": bc,
        "pm": pm,
        "ois": ois,
    }


def aggregate_scores(
    method_metrics: Mapping[str, Mapping[str, float]],
    apply_minmax: Optional[bool] = None,
) -> Dict[str, Dict[str, float]]:
    """Score each method. Min–max across methods when comparing ≥2 methods.

    method_metrics: {method_name: {map, cancer_asw, nc, sas, gpl_asw, gc}}
    """
    names = list(method_metrics.keys())
    if not names:
        return {}
    if apply_minmax is None:
        apply_minmax = len(names) >= 2

    if not apply_minmax:
        return {name: score_from_raw(method_metrics[name]) for name in names}

    normed = {name: dict(method_metrics[name]) for name in names}
    for key in list(BIO_KEYS) + list(MIX_KEYS):
        col = [float(method_metrics[n][key]) for n in names]
        scaled = minmax_normalize(col)
        for n, v in zip(names, scaled):
            normed[n][key] = v

    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        m = normed[name]
        bc = float(np.mean([m[k] for k in BIO_KEYS]))
        pm = float(np.mean([m[k] for k in MIX_KEYS]))
        ois = OIS_W_BIO * bc + OIS_W_MIX * pm
        out[name] = {
            **{k: float(method_metrics[name][k]) for k in list(BIO_KEYS) + list(MIX_KEYS)},
            **{f"{k}_norm": float(m[k]) for k in list(BIO_KEYS) + list(MIX_KEYS)},
            "bc": bc,
            "pm": pm,
            "ois": ois,
        }
    return out


def score_integration(
    x_single,
    x_integrated,
    y,
    p,
    neighbor_frac=0.01,
    n_neighbors=15,
    method_name: str = "IDFB",
) -> Dict[str, float]:
    """Score one integrated matrix with raw-scale BC/PM/OIS."""
    raw = compute_raw_metrics(
        x_single, x_integrated, y, p, neighbor_frac, n_neighbors
    )
    return score_from_raw(raw)


# Back-compat aliases
def avg_silhouette_width(x, y):
    return cancer_type_asw(x, y)


def biology_conservation(x_single, x_integrated, y, p=None, neighbor_frac=0.01):
    if p is None:
        p = pd.Series(np.zeros(len(y), dtype=int))
    raw = compute_raw_metrics(
        x_single,
        x_integrated,
        y,
        p=p,
        neighbor_frac=neighbor_frac,
    )
    # Without platform labels, only bio half is meaningful.
    return float(np.mean([raw["map"], raw["cancer_asw"], raw["nc"]]))


def gpls_mixing(x_integrated, p, y=None, neighbor_frac=0.01, n_neighbors=15):
    if y is None:
        raise ValueError("gpls_mixing requires biological labels y for GC")
    raw = compute_raw_metrics(
        x_single=x_integrated,
        x_integrated=x_integrated,
        y=y,
        p=p,
        neighbor_frac=neighbor_frac,
        n_neighbors=n_neighbors,
    )
    return float(np.mean([raw["sas"], raw["gpl_asw"], raw["gc"]]))


def run_evaluate(task: str):
    processed = DATASET_DIR / task / "processed_data.csv"
    generated = DATASET_DIR / task / "generated_data.csv"
    if not processed.exists():
        raise FileNotFoundError(f"Missing {processed}")
    if not generated.exists():
        raise FileNotFoundError(f"Missing {generated}. Run integrate first.")

    df_single = pd.read_csv(processed)
    x_single = df_single.iloc[:, :-2]
    p = df_single.iloc[:, -2]
    y = df_single.iloc[:, -1]
    x_integrated = pd.read_csv(generated)

    scores = score_integration(x_single, x_integrated, y, p, method_name="IDFB")
    print(
        f"MAP={scores['map']:.4f}  Cancer-ASW={scores['cancer_asw']:.4f}  "
        f"NC={scores['nc']:.4f}"
    )
    print(
        f"SAS={scores['sas']:.4f}  GPL-ASW={scores['gpl_asw']:.4f}  "
        f"GC={scores['gc']:.4f}"
    )
    print(
        f"BC={scores['bc']:.4f}  PM={scores['pm']:.4f}  "
        f"OIS={scores['ois']:.4f}  (OIS=0.6*BC+0.4*PM)"
    )
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate IDFB integration (BC/PM/OIS)")
    parser.add_argument("--task", type=str, default="MSI")
    args = parser.parse_args()
    run_evaluate(args.task)


if __name__ == "__main__":
    main()
