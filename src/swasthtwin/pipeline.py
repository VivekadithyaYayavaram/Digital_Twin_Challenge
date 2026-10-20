"""End-to-end steps: generate data -> train/select -> evaluate/report."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import __version__
from . import config as C
from . import data_generator as G
from . import evaluation as E
from . import features as F
from . import modeling as M


def _cfg_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _round(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(obj), indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------
def step_data(cfg: dict[str, Any] | None = None, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or C.load_config()
    data = G.generate_cohort(cfg)
    problems = G.validate_cohort(data)
    if problems:
        raise RuntimeError("synthetic data failed validation: " + "; ".join(problems))
    G.save_cohort(data, out_dir)
    print(f"[data] {len(data['ehr'])} patients, {len(data['sensor']):,} sensor rows, "
          f"{len(data['sleep'])} nights, {len(data['meal_log'])} logged meals -> {out_dir or C.DATA_DIR}")
    return data


def _xy(df: pd.DataFrame, cols: list[str] | None = None):
    return df[cols or F.ALL_FEATURES], df["label"].to_numpy()


def reference_row(train_df: pd.DataFrame) -> dict[str, Any]:
    """'Typical training patient' values used as the counterfactual reference in
    perturbation explanations: median for numeric, mode for categorical."""
    ref: dict[str, Any] = {c: float(train_df[c].median()) for c in F.NUMERIC_FEATURES}
    ref.update({c: str(train_df[c].mode().iloc[0]) for c in F.CATEGORICAL_FEATURES})
    return ref


def step_train(cfg: dict[str, Any] | None = None, data_dir: Path | None = None,
               artifact_dir: Path | None = None) -> dict[str, Any]:
    cfg = cfg or C.load_config()
    adir = artifact_dir or C.ARTIFACT_DIR
    adir.mkdir(parents=True, exist_ok=True)
    cohort = G.load_cohort(data_dir)
    df = F.build_dataset(cohort, cfg)
    split = M.split_patients(cohort["ehr"], cfg)
    split.to_csv(adir / C.SPLIT_FILE, index=False)
    df = df.merge(split, on="patient_id")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    seed = int(cfg["seed"])

    selection = {}
    fitted = {}
    for name in cfg["model"]["candidates"]:
        pipe = M.make_pipeline(name, seed).fit(*_xy(tr))
        p = pipe.predict_proba(_xy(va)[0])[:, 1]
        thr = E.threshold_for_sensitivity(va["label"].to_numpy(), p,
                                          float(cfg["model"]["operating_point"]["target_sensitivity"]))
        selection[name] = E.binary_metrics(va["label"], p, thr)
        fitted[name] = pipe
        print(f"[train] {name:24s} val PR-AUC {selection[name]['pr_auc']:.4f} "
              f"ROC-AUC {selection[name]['roc_auc']:.4f}")
    rule_val = E.binary_metrics(va["label"], M.rule_baseline_score(va, cfg["target"]["threshold_mg_dl"]), 0.5)
    metric = cfg["model"]["selection_metric"]
    best = max(selection, key=lambda n: selection[n][metric])
    print(f"[train] selected: {best} (by validation {metric})")

    joblib.dump(fitted[best], adir / C.MODEL_FILE, compress=3)
    meta = {
        "model_version": f"{__version__}+{_cfg_hash(cfg)}",
        "package_version": __version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best,
        "selected_model_label": M.MODEL_NAMES[best],
        "selection_metric": metric,
        "operating_threshold": selection[best]["threshold"],
        "operating_rule": cfg["model"]["operating_point"],
        "target": cfg["target"],
        "features": {"numeric": F.NUMERIC_FEATURES, "categorical": F.CATEGORICAL_FEATURES,
                     "stream": F.FEATURE_STREAM},
        "reference_row": reference_row(tr),
        "data": {"seed": cfg["seed"], "n_patients": cfg["data"]["n_patients"],
                 "n_days": cfg["data"]["n_days"], "synthetic": True},
        "split_counts": {k: int(v) for k, v in split["split"].value_counts().items()},
        "train_rows": int(len(tr)), "val_rows": int(len(va)),
        "config_hash": _cfg_hash(cfg),
    }
    _write_json(adir / C.MODEL_META_FILE, meta)
    _write_json(adir / "model_selection.json", {
        "split": "validation patients (never used for fitting)",
        "selection_metric": metric, "selected": best,
        "candidates": selection, "rule_baseline": rule_val})
    return meta


# ----------------------------------------------------------------------------
def _grouped_permutation(pipe, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]],
                         reps: int, seed: int) -> dict[str, dict[str, float]]:
    """Drop in PR-AUC when a whole block of columns is shuffled jointly across rows."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, pipe.predict_proba(X)[:, 1])
    out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(reps):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - average_precision_score(y, pipe.predict_proba(Xp)[:, 1]))
        out[g] = {"mean_drop_pr_auc": float(np.mean(drops)), "sd": float(np.std(drops))}
    return out


def step_evaluate(cfg: dict[str, Any] | None = None, data_dir: Path | None = None,
                  artifact_dir: Path | None = None, report_dir: Path | None = None,
                  bootstrap_reps: int | None = None, make_figures: bool = True) -> dict[str, Any]:
    from sklearn.inspection import permutation_importance

    cfg = cfg or C.load_config()
    adir, rdir = artifact_dir or C.ARTIFACT_DIR, report_dir or C.REPORT_DIR
    fdir = rdir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((adir / C.MODEL_META_FILE).read_text())
    sel = json.loads((adir / "model_selection.json").read_text())
    final = joblib.load(adir / C.MODEL_FILE)
    best = meta["selected_model"]
    seed = int(cfg["seed"])
    thr_mg = float(cfg["target"]["threshold_mg_dl"])
    reps = int(bootstrap_reps or cfg["model"]["bootstrap_reps"])
    tsens = float(cfg["model"]["operating_point"]["target_sensitivity"])

    cohort = G.load_cohort(data_dir)
    df = F.build_dataset(cohort, cfg).merge(pd.read_csv(adir / C.SPLIT_FILE), on="patient_id")
    tr, va, te = (df[df.split == s].copy() for s in ("train", "val", "test"))
    assert not (set(tr.patient_id) & set(te.patient_id)), "patient leakage between train and test"

    # ---- scores on TEST for every model --------------------------------------
    te["score_rule_baseline"] = M.rule_baseline_score(te, thr_mg)
    thresholds = {"rule_baseline": 0.5}
    for name in cfg["model"]["candidates"]:
        pipe = final if name == best else M.make_pipeline(name, seed).fit(*_xy(tr))
        te[f"score_{name}"] = pipe.predict_proba(_xy(te)[0])[:, 1]
        thresholds[name] = float(sel["candidates"][name]["threshold"])
    order = ["rule_baseline"] + list(cfg["model"]["candidates"])
    score_cols = {m: f"score_{m}" for m in order}

    test_metrics = {m: E.binary_metrics(te["label"], te[c], thresholds[m]) for m, c in score_cols.items()}
    onset = te[te["cgm_current"] < thr_mg]
    onset_metrics = {m: E.binary_metrics(onset["label"], onset[c], thresholds[m])
                     for m, c in score_cols.items()}
    boot = E.patient_bootstrap(te, score_cols, thresholds, reps, seed)
    print(f"[eval] test {best}: PR-AUC {test_metrics[best]['pr_auc']:.4f} "
          f"ROC-AUC {test_metrics[best]['roc_auc']:.4f}")

    # ---- temporal check: early days of train patients -> late days of test patients
    td = int(cfg["split"]["temporal_train_days"])
    tr_early, va_early = tr[tr.study_day < td], va[va.study_day < td]
    te_late = te[te.study_day >= td].copy()
    tpipe = M.make_pipeline(best, seed).fit(*_xy(tr_early))
    t_thr = E.threshold_for_sensitivity(va_early["label"].to_numpy(),
                                        tpipe.predict_proba(_xy(va_early)[0])[:, 1], tsens)
    te_late["s"] = tpipe.predict_proba(_xy(te_late)[0])[:, 1]
    temporal = {
        "description": f"fit on TRAIN patients, study days 0-{td - 1}; threshold from VAL patients days "
                       f"0-{td - 1}; evaluated on TEST patients, days {td}+ (new patients AND later time)",
        best: E.binary_metrics(te_late["label"], te_late["s"], t_thr),
        "rule_baseline": E.binary_metrics(te_late["label"], M.rule_baseline_score(te_late, thr_mg), 0.5),
    }

    # ---- ablation: does fusing the two streams help? ---------------------------
    sets = {
        "ehr_only": (F.EHR_NUMERIC + F.CONTEXT_FEATURES, F.EHR_CATEGORICAL),
        "sensor_only": (F.SENSOR_FEATURES + F.CONTEXT_FEATURES, []),
        "fused_ehr_plus_sensor": (F.NUMERIC_FEATURES, F.CATEGORICAL_FEATURES),
    }
    # rows where the CGM is stale or sparse (sensor gap): does the EHR help more here?
    gap = te[(te["cgm_minutes_since_last"] > float(cfg["features"]["max_staleness_min"]))
             | (te["cgm_coverage_120"] < 0.5)]
    ablation = {}
    for k, (num, cat) in sets.items():
        pipe = M.make_pipeline(best, seed, num, cat).fit(tr[num + cat], tr["label"])
        pv = pipe.predict_proba(va[num + cat])[:, 1]
        thr_k = E.threshold_for_sensitivity(va["label"].to_numpy(), pv, tsens)
        te[f"abl_{k}"] = pipe.predict_proba(te[num + cat])[:, 1]
        ablation[k] = {"test": E.binary_metrics(te["label"], te[f"abl_{k}"], thr_k),
                       "test_onset_subset": E.binary_metrics(
                           te.loc[onset.index, "label"], te.loc[onset.index, f"abl_{k}"], thr_k),
                       "test_sensor_gap_subset": E.binary_metrics(
                           te.loc[gap.index, "label"], te.loc[gap.index, f"abl_{k}"], thr_k)}
    abl_boot = E.patient_bootstrap(te, {k: f"abl_{k}" for k in sets},
                                   {k: ablation[k]["test"]["threshold"] for k in sets}, reps, seed,
                                   metrics=("pr_auc", "roc_auc"))

    # ---- importance --------------------------------------------------------
    sub = te.sample(n=min(6000, len(te)), random_state=seed)
    pim = permutation_importance(final, sub[F.ALL_FEATURES], sub["label"], scoring="average_precision",
                                 n_repeats=5, random_state=seed, n_jobs=1)
    perm = sorted(({"feature": f, "stream": F.FEATURE_STREAM[f], "mean_drop_pr_auc": float(m),
                    "sd": float(s)} for f, m, s in zip(F.ALL_FEATURES, pim.importances_mean,
                                                       pim.importances_std)),
                  key=lambda r: -r["mean_drop_pr_auc"])
    grouped = _grouped_permutation(final, sub[F.ALL_FEATURES], sub["label"].to_numpy(), {
        "EHR (historical) block": F.EHR_NUMERIC + F.EHR_CATEGORICAL,
        "Sensor (dynamic) block": F.SENSOR_FEATURES,
        "Time-of-day context": F.CONTEXT_FEATURES,
        "Fusion features": F.INTERACTION_FEATURES}, reps=5, seed=seed)

    # ---- assemble ---------------------------------------------------------------
    def counts(d):
        return {"rows": int(len(d)), "patients": int(d.patient_id.nunique()),
                "prevalence": float(d.label.mean())}
    metrics = {
        "disclaimer": "All numbers measure agreement with a SYNTHETIC data generator. They are not "
                      "evidence of clinical accuracy or benefit in real patients.",
        "model_version": meta["model_version"],
        "target": cfg["target"],
        "selected_model": best,
        "operating_point": {"rule": f"highest threshold reaching sensitivity >= {tsens} on validation "
                                    "patients", "threshold": thresholds[best]},
        "splits": {"method": "by patient (no patient in more than one split), stratified by "
                             "diabetes status; seed " + str(seed),
                   "train": counts(tr), "val": counts(va), "test": counts(te),
                   "test_onset_subset": {**counts(onset), "definition":
                                         f"test rows where latest CGM < {thr_mg:g} mg/dL"},
                   "test_sensor_gap_subset": {**counts(gap), "definition":
                                              "test rows with last CGM older than max_staleness_min "
                                              "or <50% CGM coverage in the past 120 min"}},
        "model_labels": M.MODEL_NAMES,
        "validation_selection": {m: {"pr_auc": sel["candidates"][m]["pr_auc"],
                                     "roc_auc": sel["candidates"][m]["roc_auc"]}
                                 for m in cfg["model"]["candidates"]},
        "test": test_metrics,
        "test_onset_subset": onset_metrics,
        "bootstrap_ci_test": boot,
        "temporal_holdout": temporal,
        "ablation": ablation,
        "ablation_bootstrap_ci": abl_boot,
        "permutation_importance_test": perm,
        "grouped_permutation_importance_test": grouped,
    }
    _write_json(rdir / C.METRICS_FILE, metrics)
    (rdir / "metrics.md").write_text(render_metrics_md(_round(metrics)), encoding="utf-8")

    if make_figures:
        lab = {m: M.MODEL_NAMES[m].split(" (")[0] for m in order}
        lab["rule_baseline"] = "Rule baseline"
        E.plot_roc_pr(te, score_cols, lab, fdir / "roc_pr_test.png")
        E.plot_calibration(te, {m: score_cols[m] for m in cfg["model"]["candidates"]}, lab,
                           fdir / "calibration_test.png")
        E.plot_confusion(test_metrics[best]["confusion"],
                         f"{lab[best]} — test, threshold {thresholds[best]:.2f}",
                         fdir / "confusion_test.png")
        E.plot_barh({"EHR + time": ablation["ehr_only"]["test"]["pr_auc"],
                     "Sensor + time": ablation["sensor_only"]["test"]["pr_auc"],
                     "Fused EHR + sensor": ablation["fused_ehr_plus_sensor"]["test"]["pr_auc"]},
                    "Two-stream ablation — PR-AUC on test patients", "PR-AUC (95% patient-bootstrap CI)",
                    fdir / "ablation_test.png",
                    color={"EHR + time": "#8A6FB0", "Sensor + time": "#2F5D9E", "Fused EHR + sensor": "#0F9D8F"},
                    errors={"EHR + time": abl_boot["models"]["ehr_only"]["pr_auc"],
                            "Sensor + time": abl_boot["models"]["sensor_only"]["pr_auc"],
                            "Fused EHR + sensor": abl_boot["models"]["fused_ehr_plus_sensor"]["pr_auc"]},
                    xlim=(0, 1.12))
        top = perm[:12]
        E.plot_barh({F.FEATURE_LABELS[r["feature"]]: r["mean_drop_pr_auc"] for r in top},
                    "Permutation importance — test patients", "Mean drop in PR-AUC when shuffled",
                    fdir / "permutation_importance_test.png",
                    color={F.FEATURE_LABELS[r["feature"]]:
                           ("#8A6FB0" if r["stream"].startswith("EHR") else
                            "#D08C2E" if r["stream"].startswith("Fusion") else
                            "#8A94A6" if r["stream"].startswith("Context") else "#2F5D9E") for r in top})
    print(f"[eval] reports written to {rdir}")
    return metrics


# ----------------------------------------------------------------------------
def _f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render_metrics_md(m: dict[str, Any]) -> str:
    lab = m["model_labels"]
    best = m["selected_model"]
    ci = m["bootstrap_ci_test"]["models"]
    L = ["# SwasthTwin — evaluation report", "",
         "> **Generated automatically by `make evaluate` from `reports/metrics.json`. Do not edit by hand.**",
         f"> {m['disclaimer']}", "",
         f"- Target: `{m['target']['name']}` — any CGM reading >= {m['target']['threshold_mg_dl']} mg/dL "
         f"in the next {m['target']['horizon_min']} min (prototype rule, not a clinical definition).",
         f"- Model version: `{m['model_version']}`; selected model: **{lab[best]}** "
         "(chosen on validation patients by PR-AUC).",
         f"- Operating threshold: {m['operating_point']['threshold']:.3f} "
         f"({m['operating_point']['rule']}).",
         f"- Split: {m['splits']['method']}.", "",
         "## Split sizes", "", "| Split | Rows | Patients | Event prevalence |", "|---|---:|---:|---:|"]
    for k in ("train", "val", "test", "test_onset_subset", "test_sensor_gap_subset"):
        s = m["splits"][k]
        L.append(f"| {k} | {s['rows']:,} | {s['patients']} | {_f(s['prevalence'])} |")
    L += ["", "## Validation (model selection)", "", "| Model | PR-AUC | ROC-AUC |", "|---|---:|---:|"]
    for k, v in m["validation_selection"].items():
        L.append(f"| {lab[k]} | {_f(v['pr_auc'])} | {_f(v['roc_auc'])} |")

    def table(block, title, with_ci=False):
        out = ["", f"## {title}", "",
               "| Model | ROC-AUC | PR-AUC | Brier | Threshold | Sensitivity | Specificity | Precision | F1 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in block.items():
            roc, pr = _f(v["roc_auc"]), _f(v["pr_auc"])
            if with_ci and k in ci:
                roc += f" [{_f(ci[k]['roc_auc'][0])}–{_f(ci[k]['roc_auc'][1])}]"
                pr += f" [{_f(ci[k]['pr_auc'][0])}–{_f(ci[k]['pr_auc'][1])}]"
            out.append(f"| {lab.get(k, k)} | {roc} | {pr} | {_f(v['brier'])} | {_f(v['threshold'], 2)} | "
                       f"{_f(v['sensitivity'])} | {_f(v['specificity'])} | {_f(v['precision'])} | {_f(v['f1'])} |")
        return out
    L += table(m["test"], "Held-out test patients (95% patient-bootstrap CI in brackets)", True)
    c = m["test"][best]["confusion"]
    L += ["", f"Confusion matrix ({lab[best]}, test): TP {c['tp']:,} · FP {c['fp']:,} · "
              f"TN {c['tn']:,} · FN {c['fn']:,}"]
    L += table(m["test_onset_subset"],
               "Harder subset: test rows where glucose is NOT already >= threshold")
    L += ["", "Paired bootstrap differences (selected model minus comparator, 95% CI):", ""]
    for k, v in m["bootstrap_ci_test"]["paired_differences"].items():
        if k.startswith(best + "_minus_"):
            L.append(f"- {k}: PR-AUC [{_f(v['pr_auc'][0])}, {_f(v['pr_auc'][1])}], "
                     f"ROC-AUC [{_f(v['roc_auc'][0])}, {_f(v['roc_auc'][1])}]")
    t = m["temporal_holdout"]
    L += ["", "## Temporal hold-out", "", t["description"], "",
          "| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Rows |", "|---|---:|---:|---:|---:|---:|"]
    for k in (best, "rule_baseline"):
        v = t[k]
        L.append(f"| {lab[k]} | {_f(v['roc_auc'])} | {_f(v['pr_auc'])} | {_f(v['sensitivity'])} | "
                 f"{_f(v['specificity'])} | {v['n']:,} |")
    ab, abci = m["ablation"], m["ablation_bootstrap_ci"]["models"]
    L += ["", "## Two-stream ablation (same algorithm, different inputs)", "",
          "Both single-stream models also receive time-of-day context so the comparison is fair.", "",
          "| Inputs | Test PR-AUC [95% CI] | Test ROC-AUC | Onset-subset PR-AUC | Sensor-gap-subset PR-AUC |",
          "|---|---:|---:|---:|---:|"]
    for k, v in ab.items():
        L.append(f"| {k} | {_f(v['test']['pr_auc'])} [{_f(abci[k]['pr_auc'][0])}–{_f(abci[k]['pr_auc'][1])}] | "
                 f"{_f(v['test']['roc_auc'])} | {_f(v['test_onset_subset']['pr_auc'])} | "
                 f"{_f(v['test_sensor_gap_subset']['pr_auc'])} |")
    L += ["", "## Permutation importance (test, top 12)", "",
          "Predictive association only — not causal.", "", "| Feature | Stream | Mean PR-AUC drop |",
          "|---|---|---:|"]
    for r in m["permutation_importance_test"][:12]:
        L.append(f"| `{r['feature']}` | {r['stream']} | {_f(r['mean_drop_pr_auc'], 4)} |")
    L += ["", "Grouped (whole block shuffled):", ""]
    for k, v in m["grouped_permutation_importance_test"].items():
        L.append(f"- {k}: mean PR-AUC drop {_f(v['mean_drop_pr_auc'], 4)} (sd {_f(v['sd'], 4)})")
    L += ["", "## Figures", "", "![ROC/PR](figures/roc_pr_test.png)", "",
          "![Calibration](figures/calibration_test.png)", "", "![Confusion](figures/confusion_test.png)", "",
          "![Ablation](figures/ablation_test.png)", "", "![Importance](figures/permutation_importance_test.png)", ""]
    return "\n".join(L)
