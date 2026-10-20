"""Metrics, operating-threshold selection, patient-level bootstrap and figures."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score)

# Validated categorical palette (see docs/design_system.md). Fixed per model.
MODEL_COLORS = {
    "hist_gradient_boosting": "#0F9D8F",
    "logistic_regression": "#2F5D9E",
    "random_forest": "#D08C2E",
    "rule_baseline": "#8A94A6",
}
NAVY = "#0B1F3A"


def _safe(fn, *a) -> float | None:
    try:
        v = float(fn(*a))
        return None if np.isnan(v) else v
    except ValueError:
        return None


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    div = lambda a, b: (a / b) if b else None  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if prec is not None else None)
    two = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "n_positive": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "roc_auc": _safe(roc_auc_score, y, p) if two else None,
        "pr_auc": _safe(average_precision_score, y, p) if two else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "threshold": float(threshold),
        "precision": prec, "sensitivity": rec, "specificity": div(tn, tn + fp),
        "f1": f1, "accuracy": div(tp + tn, len(y)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity on (y, p) is still >= target."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def patient_bootstrap(df: pd.DataFrame, score_cols: dict[str, str], thresholds: dict[str, float],
                      reps: int, seed: int,
                      metrics=("roc_auc", "pr_auc", "brier", "sensitivity", "specificity")
                      ) -> dict[str, Any]:
    """Cluster bootstrap: resample PATIENTS with replacement (rows of one patient
    are correlated, so row-level resampling would give over-narrow intervals).
    Returns 95% percentile intervals per model and for paired differences vs
    every other model."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)  # positional indices below
    groups = {pid: idx.to_numpy() for pid, idx in df.groupby("patient_id").groups.items()}
    pids = np.array(sorted(groups))
    y_all = df["label"].to_numpy()
    draws: dict[str, dict[str, list[float]]] = {m: {k: [] for k in metrics} for m in score_cols}
    for _ in range(reps):
        pick = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([groups[p] for p in pick])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue
        for m, col in score_cols.items():
            res = binary_metrics(y, df[col].to_numpy()[idx], thresholds[m])
            for k in metrics:
                if res[k] is not None:
                    draws[m][k].append(res[k])
    out: dict[str, Any] = {"method": f"patient-level cluster bootstrap, {reps} resamples, "
                                     "95% percentile intervals", "models": {}}
    for m in score_cols:
        out["models"][m] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                            for k, v in draws[m].items() if v}
    names = list(score_cols)
    out["paired_differences"] = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            n = min(len(draws[a].get("pr_auc", [])), len(draws[b].get("pr_auc", [])),
                    len(draws[a].get("roc_auc", [])), len(draws[b].get("roc_auc", [])))
            if n == 0:
                continue
            d_pr = np.array(draws[a]["pr_auc"][:n]) - np.array(draws[b]["pr_auc"][:n])
            d_roc = np.array(draws[a]["roc_auc"][:n]) - np.array(draws[b]["roc_auc"][:n])
            out["paired_differences"][f"{a}_minus_{b}"] = {
                "pr_auc": [float(np.percentile(d_pr, 2.5)), float(np.percentile(d_pr, 97.5))],
                "roc_auc": [float(np.percentile(d_roc, 2.5)), float(np.percentile(d_roc, 97.5))],
            }
    return out


# ----------------------------------------------------------------------------
# Figures (matplotlib, static PNG for reports and slides)
# ----------------------------------------------------------------------------

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9D1DC")
    ax.tick_params(colors="#44536A", labelsize=9)
    ax.grid(True, color="#E8EDF3", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc_pr(df: pd.DataFrame, score_cols: dict[str, str], labels: dict[str, str], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    y = df["label"].to_numpy()
    for m, col in score_cols.items():
        p = df[col].to_numpy()
        ls = "--" if m == "rule_baseline" else "-"
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, ls, lw=2, color=MODEL_COLORS[m],
                     label=f"{labels[m]} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, ls, lw=2, color=MODEL_COLORS[m],
                     label=f"{labels[m]} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="#B8C2CF", lw=1)
    axes[1].axhline(y.mean(), ls=":", color="#B8C2CF", lw=1)
    axes[1].text(0.99, y.mean() + 0.012, f"prevalence {y.mean():.2f}", ha="right", color="#6B7A90", fontsize=8)
    for ax, (xl, yl, t) in zip(axes, [("False positive rate", "Sensitivity", "ROC curve"),
                                      ("Sensitivity (recall)", "Precision", "Precision-recall curve")]):
        _style(ax); ax.set_xlabel(xl, color=NAVY); ax.set_ylabel(yl, color=NAVY)
        ax.set_title(t, color=NAVY, fontsize=11, loc="left", fontweight="bold")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right" if ax is axes[0] else "lower left")
    fig.suptitle("Held-out TEST patients (synthetic cohort)", x=0.01, ha="left", color="#44536A", fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_calibration(df: pd.DataFrame, score_cols: dict[str, str], labels: dict[str, str], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], ":", color="#B8C2CF", lw=1, label="Perfect calibration")
    for m, col in score_cols.items():
        fr, mp = calibration_curve(df["label"], df[col], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "-o", ms=4.5, lw=2, color=MODEL_COLORS[m], label=labels[m])
    _style(ax)
    ax.set_xlabel("Predicted probability (decile mean)", color=NAVY)
    ax.set_ylabel("Observed event rate", color=NAVY)
    ax.set_title("Calibration — test patients", color=NAVY, fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_confusion(conf: dict[str, int], title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.9), dpi=150)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=13, color=NAVY,
                    fontweight="bold")
    ax.set_xticks([0, 1], ["Predicted no event", "Predicted event"], fontsize=9)
    ax.set_yticks([0, 1], ["Actual no event", "Actual event"], fontsize=9)
    ax.set_title(title, color=NAVY, fontsize=10, loc="left", fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_barh(values: dict[str, float], title: str, xlabel: str, path, color="#0F9D8F",
              errors: dict[str, list[float]] | None = None, xlim=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(values.items())[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(items) + 1.2), dpi=150)
    names = [k for k, _ in items]; vals = [v for _, v in items]
    cols = color if isinstance(color, str) else [color[k] for k in names]
    ax.barh(names, vals, color=cols, height=0.62)
    if errors:
        for i, k in enumerate(names):
            if k in errors:
                lo, hi = errors[k]
                ax.plot([lo, hi], [i, i], color=NAVY, lw=1.4)
    for i, (k, v) in enumerate(zip(names, vals)):
        x = max(v, errors[k][1]) if errors and k in errors else v
        ax.text(x, i, f"  {v:.3f}", va="center", fontsize=8.5, color=NAVY)
    _style(ax); ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=NAVY)
    ax.set_title(title, color=NAVY, fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
