"""DEG overlap for integration methods.

Reference: genes that are DE (Welch t-test, BH q<0.05) in at least two
platforms on uncorrected data. Each method is scored by Jaccard of its
pooled DEG set against that reference.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _binary_labels(y: pd.Series) -> Optional[Tuple[np.ndarray, object, object]]:
    vals = pd.Series(y).reset_index(drop=True)
    uniq = list(vals.unique())
    if len(uniq) < 2:
        return None
    # Use the two most frequent classes for a stable comparison.
    counts = vals.value_counts()
    a, b = counts.index[0], counts.index[1]
    mask = vals.isin([a, b]).to_numpy()
    lab = (vals.to_numpy() == a)
    return mask, a, b, lab


def deg_genes(
    X: np.ndarray,
    y,
    q_cutoff: float = 0.05,
    min_n: int = 8,
) -> Set[int]:
    """Return gene indices DE between the two largest biological classes."""
    parsed = _binary_labels(y)
    if parsed is None:
        return set()
    mask, _a, _b, lab = parsed
    X = np.asarray(X)[mask]
    lab = lab[mask]
    n0 = int((~lab).sum())
    n1 = int(lab.sum())
    if n0 < min_n or n1 < min_n:
        return set()
    t, p = stats.ttest_ind(X[lab], X[~lab], axis=0, equal_var=False, nan_policy="omit")
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    q = bh_fdr(p)
    return set(np.where(q < q_cutoff)[0].tolist())


def platform_reference_degs(
    X: np.ndarray,
    y,
    p,
    q_cutoff: float = 0.05,
    min_platforms: int = 2,
) -> Set[int]:
    """Genes DE on uncorrected data in >= min_platforms platforms."""
    p = pd.Series(p).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    hits: Dict[int, int] = {}
    for plat in p.unique():
        mask = (p == plat).to_numpy()
        genes = deg_genes(X[mask], y[mask], q_cutoff=q_cutoff)
        for g in genes:
            hits[g] = hits.get(g, 0) + 1
    return {g for g, c in hits.items() if c >= min_platforms}


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deg_report(
    X_raw: np.ndarray,
    y,
    p,
    method_matrices: Dict[str, np.ndarray],
    q_cutoff: float = 0.05,
) -> pd.DataFrame:
    ref = platform_reference_degs(X_raw, y, p, q_cutoff=q_cutoff)
    n_ref = len(ref)
    rows = []
    deg_sets = {}
    for name, Xm in method_matrices.items():
        genes = deg_genes(Xm, y, q_cutoff=q_cutoff)
        deg_sets[name] = genes
        overlap = len(genes & ref)
        rows.append(
            {
                "method": name,
                "n_deg": int(len(genes)),
                "n_ref_deg": n_ref,
                "overlap_ref": overlap,
                "jaccard_ref": jaccard(genes, ref),
                "recall_ref": (overlap / n_ref) if n_ref else 0.0,
            }
        )
    if "IDFB" in deg_sets:
        idfb = deg_sets["IDFB"]
        for row in rows:
            row["jaccard_vs_IDFB"] = jaccard(deg_sets[row["method"]], idfb)
    return pd.DataFrame(rows)
