"""Compare a leaky test with an honest one.

Leaky: pick the measurements using everyone, then hide people and score.
Honest: hide people first, then pick the measurements.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 99
N_SAMPLES = 80
N_FEATURES = 400
N_INFORMATIVE = 8
N_SELECT = 20
N_SPLITS = 5
MIN_ACCURACY_GAP = 0.15
FIGURES = Path("figures")
CLASS_COLOUR = {0: "#d17a22", 1: "#5b3a8c"}
METRICS = ("accuracy", "precision", "recall", "f1")


def make_table(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=0.8,
        flip_y=0.05,
        random_state=seed,
    )


def new_splits(seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


def new_svm(seed: int = SEED) -> SVC:
    return SVC(C=1, kernel="linear", random_state=seed)


def preprocess() -> list[tuple[str, object]]:
    return [
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=N_SELECT)),
    ]


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def means(folds: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([fold[name] for fold in folds])) for name in METRICS}


def true_signal_count(support: np.ndarray) -> int:
    return int(support[:N_INFORMATIVE].sum())


def evaluate_leaky(X: np.ndarray, y: np.ndarray) -> dict:
    """Pick measurements using everyone, then hide people and score."""
    transform = Pipeline(preprocess())
    X_selected = transform.fit_transform(X, y)
    folds = []
    svm = new_svm()
    for train_idx, test_idx in new_splits().split(X_selected, y):
        svm.fit(X_selected[train_idx], y[train_idx])
        folds.append(score(y[test_idx], svm.predict(X_selected[test_idx])))
    return {
        "folds": folds,
        "means": means(folds),
        "X_selected": X_selected,
        "n_true_selected": true_signal_count(
            transform.named_steps["select"].get_support()
        ),
    }


def evaluate_honest(X: np.ndarray, y: np.ndarray) -> dict:
    """Hide people first, then pick measurements."""
    pipeline = Pipeline([*preprocess(), ("svm", new_svm())])
    folds = []
    first_round = None
    for train_idx, test_idx in new_splits().split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        folds.append(score(y[test_idx], pipeline.predict(X[test_idx])))
        if first_round is None:
            support = pipeline.named_steps["select"].get_support()
            scaler = pipeline.named_steps["scale"]
            first_round = {
                "X_train": scaler.transform(X[train_idx])[:, support],
                "X_test": scaler.transform(X[test_idx])[:, support],
                "y_train": y[train_idx],
                "y_test": y[test_idx],
                "n_true_selected": true_signal_count(support),
            }
    return {
        "folds": folds,
        "means": means(folds),
        "first_round": first_round,
    }


def scatter_classes(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    marker: str,
    prefix: str = "",
) -> None:
    for cls, colour in CLASS_COLOUR.items():
        mask = labels == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colour,
            edgecolors="0.2",
            linewidths=0.4 if marker == "o" else 0.8,
            alpha=0.85,
            marker=marker,
            s=36 if marker == "o" else 70,
            label=f"{prefix}class {cls}",
        )


def plot_leaky_pca(X_selected: np.ndarray, y: np.ndarray, path: Path) -> None:
    coords = PCA(n_components=2, random_state=SEED).fit_transform(X_selected)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter_classes(ax, coords, y, marker="o")
    ax.set_title("Measurements chosen using everyone")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_honest_pca(first_round: dict, path: Path) -> None:
    pca = PCA(n_components=2, random_state=SEED).fit(first_round["X_train"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter_classes(
        ax,
        pca.transform(first_round["X_train"]),
        first_round["y_train"],
        "o",
        "used to pick, ",
    )
    scatter_classes(
        ax,
        pca.transform(first_round["X_test"]),
        first_round["y_test"],
        "^",
        "hidden, ",
    )
    ax.set_title("Measurements chosen after hiding some people")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metrics(leaky: dict, honest: dict, path: Path) -> None:
    x = np.arange(len(METRICS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        x - width / 2,
        [leaky["means"][name] for name in METRICS],
        width,
        label="leaky",
        color="#c0392b",
    )
    ax.bar(
        x + width / 2,
        [honest["means"][name] for name in METRICS],
        width,
        label="honest",
        color="#1f6f4a",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Average over five tests")
    ax.set_title("Same model. The leaky version peeked.")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_report(leaky: dict, honest: dict) -> None:
    print(f"{'metric':<12} {'leaky':>8} {'honest':>8} {'gap':>8}")
    for name in METRICS:
        leaked = leaky["means"][name]
        fair = honest["means"][name]
        print(f"{name:<12} {leaked:8.3f} {fair:8.3f} {leaked - fair:8.3f}")
    print()
    print(
        f"Real measurements in the leaky top {N_SELECT}: "
        f"{leaky['n_true_selected']} of {N_INFORMATIVE}"
    )
    print(
        f"Real measurements in the first honest pick: "
        f"{honest['first_round']['n_true_selected']} of {N_INFORMATIVE}"
    )
    print()
    print("Accuracy on each hidden group")
    print("  leaky   " + "  ".join(f"{fold['accuracy']:.3f}" for fold in leaky["folds"]))
    print("  honest  " + "  ".join(f"{fold['accuracy']:.3f}" for fold in honest["folds"]))


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    X, y = make_table()
    leaky = evaluate_leaky(X, y)
    honest = evaluate_honest(X, y)
    gap = leaky["means"]["accuracy"] - honest["means"]["accuracy"]
    if gap < MIN_ACCURACY_GAP:
        raise SystemExit(
            f"expected the leaky score to beat the honest score by "
            f"{MIN_ACCURACY_GAP:.2f}, got {gap:.3f}"
        )

    plot_leaky_pca(leaky["X_selected"], y, FIGURES / "pca_leaky.png")
    plot_honest_pca(honest["first_round"], FIGURES / "pca_honest.png")
    plot_metrics(leaky, honest, FIGURES / "metrics.png")
    print_report(leaky, honest)
    print()
    print(f"Wrote {FIGURES / 'metrics.png'}")
    print(f"Wrote {FIGURES / 'pca_leaky.png'}")
    print(f"Wrote {FIGURES / 'pca_honest.png'}")


if __name__ == "__main__":
    main()
