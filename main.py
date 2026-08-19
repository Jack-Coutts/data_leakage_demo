"""Compare a cheating test with an honest one.

The cheat: pick the measurements using everyone, then hide people and score.
Honest: hide people first, then pick the measurements from whoever is left.

Everything else about the two runs is identical, so any difference in the
scores comes from that one change.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
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
METRICS = ("accuracy", "precision", "recall", "f1")

# What the two groups of people are, in the story we tell around the numbers.
GROUP_NAME = {0: "no disease", 1: "has the disease"}

# Categorical hues, checked for colour-blind separation against a light
# background. Never a red/green pair.
GROUP_COLOUR = {0: "#2a78d6", 1: "#eb6834"}
CHEAT_COLOUR = "#e34948"
HONEST_COLOUR = "#2a78d6"

INK = "#0b0b0b"
MUTED = "#52514e"


def make_table(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """80 people, 400 measurements each, only 8 of which mean anything."""
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
    """How many of the 8 measurements that actually matter got picked."""
    return int(support[:N_INFORMATIVE].sum())


def evaluate_cheat(X: np.ndarray, y: np.ndarray) -> dict:
    """Pick measurements using everyone, then hide people and score."""
    transform = Pipeline(preprocess())
    X_selected = transform.fit_transform(X, y)
    folds = []
    distances = np.empty(len(y))
    svm = new_svm()
    for train_idx, test_idx in new_splits().split(X_selected, y):
        svm.fit(X_selected[train_idx], y[train_idx])
        folds.append(score(y[test_idx], svm.predict(X_selected[test_idx])))
        distances[test_idx] = svm.decision_function(X_selected[test_idx])
    return {
        "folds": folds,
        "means": means(folds),
        "distances": distances,
        "n_true_selected": true_signal_count(
            transform.named_steps["select"].get_support()
        ),
    }


def evaluate_honest(X: np.ndarray, y: np.ndarray) -> dict:
    """Hide people first, then pick measurements from whoever is left."""
    pipeline = Pipeline([*preprocess(), ("svm", new_svm())])
    folds = []
    distances = np.empty(len(y))
    picked = []
    for train_idx, test_idx in new_splits().split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        folds.append(score(y[test_idx], pipeline.predict(X[test_idx])))
        distances[test_idx] = pipeline.decision_function(X[test_idx])
        picked.append(true_signal_count(pipeline.named_steps["select"].get_support()))
    return {
        "folds": folds,
        "means": means(folds),
        "distances": distances,
        "n_true_selected": picked,
    }


def tidy(ax: plt.Axes) -> None:
    """Push the frame into the background so the data reads first."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=MUTED, length=0)


def plot_one_strip(ax: plt.Axes, distances: np.ndarray, y: np.ndarray, title: str,
                   accuracy: float) -> None:
    """One row of dots per group, laid out by how the model scored each person.

    Left of the dividing line the model says no disease, right of it says
    disease. If the rows sit on their own side, the model is telling the two
    groups apart. If both rows straddle the line, it is guessing.
    """
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    rng = np.random.default_rng(SEED)
    for row, (cls, colour) in enumerate(GROUP_COLOUR.items()):
        mask = y == cls
        ax.scatter(
            distances[mask],
            np.full(mask.sum(), row) + rng.uniform(-0.13, 0.13, mask.sum()),
            c=colour,
            s=46,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
    ax.set_yticks([0, 1])
    ax.set_yticklabels([GROUP_NAME[0], GROUP_NAME[1]])
    ax.set_ylim(-0.75, 1.6)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.text(
        1.0,
        1.02,
        f"{accuracy:.0%} right",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11.5,
        color=MUTED,
    )
    # The raw score is meaningless to the reader. Which side of the line a
    # person lands on is the whole point, so only the line is labelled.
    ax.set_xticks([])
    tidy(ax)
    ax.grid(False)


def plot_scores(cheat: dict, honest: dict, y: np.ndarray, path: Path) -> None:
    limit = 1.15 * max(np.abs(cheat["distances"]).max(), np.abs(honest["distances"]).max())
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
    plot_one_strip(
        axes[0],
        cheat["distances"],
        y,
        "The cheat: measurements picked using everyone",
        cheat["means"]["accuracy"],
    )
    plot_one_strip(
        axes[1],
        honest["distances"],
        y,
        "Done properly: measurements picked after hiding people",
        honest["means"]["accuracy"],
    )
    axes[1].set_xlim(-limit, limit)
    axes[1].text(
        -limit * 0.99, -0.95, "◀ model says no disease", fontsize=10, color=MUTED
    )
    axes[1].text(
        limit * 0.99,
        -0.95,
        "model says disease ▶",
        fontsize=10,
        color=MUTED,
        ha="right",
    )
    axes[1].text(
        0, -0.95, "the model's dividing line", fontsize=10, color=MUTED, ha="center"
    )
    fig.suptitle(
        "Every person, scored by a model that never saw them",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=13.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_accuracy(cheat: dict, honest: dict, path: Path) -> None:
    labels = ["The cheat", "Done properly"]
    values = [cheat["means"]["accuracy"], honest["means"]["accuracy"]]
    colours = [CHEAT_COLOUR, HONEST_COLOUR]
    runs = [
        [fold["accuracy"] for fold in cheat["folds"]],
        [fold["accuracy"] for fold in honest["folds"]],
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    positions = [0, 1]
    bars = ax.bar(positions, values, width=0.46, color=colours, zorder=2)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(0.5, 0.515, "a coin flip", fontsize=10, color=MUTED, ha="center")

    rng = np.random.default_rng(SEED)
    for position, fold_scores in enumerate(runs):
        ax.scatter(
            position + rng.uniform(-0.08, 0.08, len(fold_scores)),
            fold_scores,
            s=28,
            facecolors="white",
            edgecolors=MUTED,
            linewidths=1.1,
            zorder=4,
        )
        # Clear the bar top and the highest run, whichever is higher.
        headroom = max(values[position], max(fold_scores)) + 0.045
        ax.text(
            position,
            headroom,
            f"{values[position]:.0%}",
            ha="center",
            fontsize=17,
            color=INK,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("people the model got right", color=MUTED)
    ax.set_title(
        "Same model, same people. Only the cheat peeked.",
        loc="left",
        fontsize=13.5,
        color=INK,
        pad=26,
    )
    ax.text(
        0,
        1.025,
        "hollow dots are the five separate tests",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )
    ax.tick_params(axis="x", labelsize=12)
    tidy(ax)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def print_report(cheat: dict, honest: dict) -> None:
    print(f"{'metric':<12} {'cheat':>8} {'honest':>8} {'gap':>8}")
    for name in METRICS:
        cheated = cheat["means"][name]
        fair = honest["means"][name]
        print(f"{name:<12} {cheated:8.3f} {fair:8.3f} {cheated - fair:8.3f}")
    print()
    print("Accuracy on each hidden group")
    print("  cheat   " + "  ".join(f"{fold['accuracy']:.3f}" for fold in cheat["folds"]))
    print("  honest  " + "  ".join(f"{fold['accuracy']:.3f}" for fold in honest["folds"]))
    print()
    print(f"Of the {N_INFORMATIVE} measurements that actually matter:")
    print(f"  the cheat's top {N_SELECT} kept {cheat['n_true_selected']}")
    kept = ", ".join(str(count) for count in honest["n_true_selected"])
    print(f"  the five honest picks kept {kept}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    X, y = make_table()
    cheat = evaluate_cheat(X, y)
    honest = evaluate_honest(X, y)
    gap = cheat["means"]["accuracy"] - honest["means"]["accuracy"]
    if gap < MIN_ACCURACY_GAP:
        raise SystemExit(
            f"expected the cheat to beat the honest score by "
            f"{MIN_ACCURACY_GAP:.2f}, got {gap:.3f}"
        )

    plot_accuracy(cheat, honest, FIGURES / "accuracy.png")
    plot_scores(cheat, honest, y, FIGURES / "scores.png")
    print_report(cheat, honest)
    print()
    print(f"Wrote {FIGURES / 'accuracy.png'}")
    print(f"Wrote {FIGURES / 'scores.png'}")


if __name__ == "__main__":
    main()
