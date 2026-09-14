from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from idfb.config import GATE_PROBE_PCA_DIM, GATE_PROBE_TEST_SIZE, N_GPLS, SEED


def prediction_entropy(proba: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(proba, eps, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return float(np.mean(ent))


def train_fresh_probe(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = GATE_PROBE_TEST_SIZE,
    seed: int = SEED,
    use_pca: bool = True,
    pca_dim: int = GATE_PROBE_PCA_DIM,
) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    steps = [("scaler", StandardScaler())]
    if use_pca and features.shape[1] > pca_dim:
        steps.append(
            ("pca", PCA(n_components=min(pca_dim, x_train.shape[0] - 1, features.shape[1])))
        )
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                n_jobs=1,
            ),
        )
    )
    pipe = Pipeline(steps)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    proba = pipe.predict_proba(x_test)

    ba = float(balanced_accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro"))
    ent = prediction_entropy(proba)
    try:
        nll = float(log_loss(y_test, proba, labels=list(range(int(labels.max()) + 1))))
    except Exception:
        nll = float("nan")

    chance = 1.0 / max(len(np.unique(labels)), N_GPLS)
    return {
        "balanced_accuracy": ba,
        "macro_f1": f1,
        "mean_entropy": ent,
        "log_loss": nll,
        "chance_level": chance,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
