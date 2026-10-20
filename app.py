"""SwasthTwin — clinician dashboard (Streamlit).

Run:  streamlit run app.py      (or `make app`)

Research and demonstration prototype on SYNTHETIC data. Not a medical device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from swasthtwin import charts as CH  # noqa: E402
from swasthtwin import config as C  # noqa: E402
from swasthtwin import data_generator as G  # noqa: E402
from swasthtwin import twin as T  # noqa: E402
from swasthtwin.features import FEATURE_LABELS, FEATURE_STREAM  # noqa: E402

st.set_page_config(page_title="SwasthTwin · Digital Twin", page_icon="🩺", layout="wide")

CSS = """
<style>
.block-container {padding-top: 3.2rem; max-width: 1400px;}
h1, h2, h3 {color: #0B1F3A; letter-spacing: -0.01em;}
.st-banner {background: #0B1F3A; color: #fff; padding: 14px 20px; border-radius: 12px; margin-bottom: 10px;}
.st-banner .t {font-size: 1.45rem; font-weight: 700;}
.st-banner .s {color: #9FD9D2; font-size: 0.92rem; margin-top: 2px;}
.notice {background: #FFF7E8; border: 1px solid #F0D49B; color: #5A4210; padding: 8px 14px;
         border-radius: 10px; font-size: 0.86rem; margin-bottom: 12px;}
.card {background: #fff; border: 1px solid #E3E8EF; border-radius: 12px; padding: 14px 16px; height: 100%;}
.card .k {color: #6B7A90; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;}
.card .v {color: #0B1F3A; font-size: 1.55rem; font-weight: 700; margin-top: 2px;}
.card .d {color: #6B7A90; font-size: 0.8rem;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.pill.Low {background:#E3F4EC; color:#1E8E5A;} .pill.Moderate {background:#FCEFD9; color:#9A5B00;}
.pill.Elevated {background:#FBE4E1; color:#A93226;}
.tag-ehr {background:#EFE9F7; color:#5E4691; padding:1px 8px; border-radius:6px; font-size:.75rem; font-weight:600;}
.tag-sen {background:#E5EEF9; color:#24497D; padding:1px 8px; border-radius:6px; font-size:.75rem; font-weight:600;}
.tag-sim {background:#FCEFD9; color:#9A5B00; padding:1px 8px; border-radius:6px; font-size:.75rem; font-weight:600;}
.small {color:#6B7A90; font-size:.82rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DISCLAIMER = ("⚠️ <b>Research &amp; demonstration prototype — synthetic data only.</b> Not a medical device, "
              "not validated in real patients, and not a substitute for clinical judgement. It does not "
              "diagnose, does not make treatment decisions and gives no medication or dosing advice.")


def card(k: str, v: str, d: str = "") -> str:
    return f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading synthetic cohort…")
def load_cohort():
    return G.load_cohort()


@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    return T.load_model()


def load_metrics() -> dict | None:
    p = C.REPORT_DIR / C.METRICS_FILE
    return json.loads(p.read_text()) if p.exists() else None


cfg = C.load_config()
team = C.load_team_info()

st.markdown(f"""<div class="st-banner"><div class="t">SwasthTwin</div>
<div class="s">Personalised Digital Twin · 2-hour hyperglycemia risk forecasting · {team.get('competition', '')}</div>
</div><div class="notice">{DISCLAIMER}</div>""", unsafe_allow_html=True)

try:
    cohort = load_cohort()
    pipe, meta = load_model()
except FileNotFoundError as exc:
    st.error(f"**Setup needed.** {exc}")
    st.markdown("Run these once from the project folder, then reload this page:\n\n"
                "```bash\nmake data\nmake train\nmake evaluate\n```")
    if st.button("Generate synthetic data and train the model now (~1 min)"):
        from swasthtwin import pipeline as P
        with st.spinner("Generating data and training…"):
            P.step_data(); P.step_train()
        st.cache_resource.clear()
        st.rerun()
    st.stop()

split_path = C.ARTIFACT_DIR / C.SPLIT_FILE
split = pd.read_csv(split_path) if split_path.exists() else pd.DataFrame(columns=["patient_id", "split"])
split_of = dict(zip(split.patient_id, split.split))
ehr_all = cohort["ehr"]

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Patient")
    only_test = st.toggle("Held-out test patients only", value=True,
                          help="Test patients were never seen during training, so their forecasts are "
                               "out-of-sample.")
    ids = [p for p in ehr_all.patient_id if (split_of.get(p) == "test" or not only_test)]
    if not ids:
        ids = list(ehr_all.patient_id)

    def fmt(pid):
        r = ehr_all[ehr_all.patient_id == pid].iloc[0]
        status = "T2D" if r.diabetes_status == "type2_diabetes" else "Prediabetes"
        return f"{pid} · {status} · HbA1c {r.hba1c_pct}% · {split_of.get(pid, '?')}"

    pid = st.selectbox("Synthetic patient", ids, format_func=fmt, key="pid")
    st.caption("All patients are fictional and generated by `make data`.")

    if "twins" not in st.session_state:
        st.session_state.twins = {}
    if pid not in st.session_state.twins:
        tw = T.PatientTwin(pid, cohort, pipe, meta, cfg)
        tw.refresh()
        st.session_state.twins[pid] = tw
    twin: T.PatientTwin = st.session_state.twins[pid]

    st.markdown("### Twin clock")
    st.markdown(f"**{twin.cursor:%a %d %b %Y, %H:%M}**")
    c1, c2 = st.columns(2)
    if c1.button("▶ +30 min", width="stretch", disabled=twin.at_end):
        twin.advance(30)
        st.rerun()  # re-render so every panel (incl. this clock) shows the new state
    if c2.button("⏩ +2 h", width="stretch", disabled=twin.at_end):
        for _ in range(4):
            if not twin.at_end:
                twin.advance(30)
        st.rerun()
    if st.button("↺ Reset this patient's twin", width="stretch"):
        st.session_state.twins.pop(pid, None)
        st.rerun()
    st.caption("Each step replays the next part of the synthetic sensor stream, recomputes features "
               "and re-runs the trained model.")
    st.markdown("---")
    st.markdown(f"**Model** `{meta['model_version']}`  \n{meta['selected_model_label']}  \n"
                f"Alert threshold {meta['operating_threshold']:.2f}")

fc = twin.latest
ehr = twin.ehr
obs = twin.observed_sensor()
tc = cfg["target"]

tabs = st.tabs(["Overview", "Patient timeline", "Two-stream fusion", "Prediction & explanation",
                "Digital Twin simulation", "Model evaluation", "Responsible use"])

# --------------------------------------------------------------------------- A overview
with tabs[0]:
    st.subheader(f"Patient {pid}")
    cols = st.columns(5)
    cols[0].markdown(card("120-min event probability", f"{fc.probability:.0%}",
                          f'<span class="pill {fc.risk_band}">{fc.risk_band} (prototype band)</span>'),
                     unsafe_allow_html=True)
    cols[1].markdown(card("Latest CGM", f"{fc.latest_cgm:.0f} mg/dL" if fc.latest_cgm else "—",
                          f"at {fc.latest_cgm_time:%H:%M}" if fc.latest_cgm_time is not None else "no reading"),
                     unsafe_allow_html=True)
    cols[2].markdown(card("Above model alert threshold", "Yes" if fc.above_alert_threshold else "No",
                          f"threshold {twin.threshold:.2f}"), unsafe_allow_html=True)
    cols[3].markdown(card("Twin updated", f"{fc.as_of:%H:%M}", f"{fc.as_of:%a %d %b} · source: {fc.source}"),
                     unsafe_allow_html=True)
    cols[4].markdown(card("Forecasts made", str(len(twin.history)), f"model {meta['selected_model']}"),
                     unsafe_allow_html=True)
    st.markdown(f"**Prediction target:** `{tc['name']}` — will any CGM reading reach ≥ "
                f"{tc['threshold_mg_dl']} mg/dL in the next {tc['horizon_min']} minutes? "
                "Prototype rule, not a clinical definition.")
    for w in fc.warnings:
        st.warning(w)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('#### EHR summary <span class="tag-ehr">historical · static</span>', unsafe_allow_html=True)
        e = pd.DataFrame({
            "Field": ["Age", "Sex", "BMI", "Diabetes status", "Years since diagnosis", "HbA1c",
                      "Fasting glucose", "eGFR", "LDL", "Triglycerides", "Systolic BP", "Hypertension",
                      "Medication class (recorded)"],
            "Value": [f"{ehr['age_years']} y", ehr["sex"], f"{ehr['bmi_kg_m2']} kg/m²",
                      ehr["diabetes_status"].replace("_", " "), f"{ehr['years_since_diagnosis']}",
                      f"{ehr['hba1c_pct']} %", f"{ehr['fasting_glucose_mg_dl']} mg/dL",
                      f"{ehr['egfr_ml_min_1_73m2']} mL/min/1.73m²", f"{ehr['ldl_mg_dl']} mg/dL",
                      f"{ehr['triglycerides_mg_dl']} mg/dL", f"{ehr['systolic_bp_mm_hg']} mmHg",
                      "Yes" if ehr["hypertension"] else "No", ehr["medication_class"].replace("_", " ")]})
        st.dataframe(e, hide_index=True, width="stretch", height=490)
    with right:
        st.markdown('#### Current twin state <span class="tag-sen">dynamic</span>', unsafe_allow_html=True)
        f = twin.latest_features.iloc[0]
        s = pd.DataFrame({"Signal": ["CGM trend (30 min)", "CGM mean (2 h)", "Time in 70–180 (2 h)",
                                     "Heart rate (30 min)", "HRV vs own 24 h", "Steps (2 h)",
                                     "Last night's sleep", "Logged carbs (2 h)"],
                          "Value": [f"{f.cgm_slope_30:+.2f} mg/dL/min" if pd.notna(f.cgm_slope_30) else "—",
                                    f"{f.cgm_mean_120:.0f} mg/dL" if pd.notna(f.cgm_mean_120) else "—",
                                    f"{f.cgm_tir_120:.0%}" if pd.notna(f.cgm_tir_120) else "—",
                                    f"{f.hr_mean_30:.0f} bpm" if pd.notna(f.hr_mean_30) else "—",
                                    f"{f.hrv_rel_24h:.2f}×" if pd.notna(f.hrv_rel_24h) else "—",
                                    f"{f.steps_120:,.0f}" if pd.notna(f.steps_120) else "—",
                                    f"{f.sleep_last_h:.1f} h (quality {f.sleep_quality_last:.0f})"
                                    if pd.notna(f.sleep_last_h) else "—",
                                    f"{f.carbs_logged_120:.0f} g"]})
        st.dataframe(s, hide_index=True, width="stretch")
        st.plotly_chart(CH.risk_history(twin.history_frame(), twin.threshold, cfg["risk_bands"]),
                        width="stretch", key="chart_1")

# --------------------------------------------------------------------------- B timeline
with tabs[1]:
    c1, c2 = st.columns([1, 1])
    lookback = c1.slider("Hours of history to show", 3, 48, 12, key="lookback")
    reveal = c2.toggle("Reveal what happened next (replay only — hidden from the model)", value=False,
                       help="Possible only because this is a replay of synthetic data. Shown for teaching; "
                            "the forecast was made without it.")
    st.plotly_chart(CH.cgm_timeline(obs, twin.cursor, tc["horizon_min"], tc["threshold_mg_dl"], fc.probability,
                                    fc.risk_band, lookback, twin.revealed_future() if reveal else None),
                    width="stretch", key="chart_2")
    st.caption("Solid line = observed CGM. Shaded area = the window being forecast; no glucose values are "
               "projected into it. Gaps = sensor dropouts.")
    a, b = st.columns(2)
    a.plotly_chart(CH.single_series(obs, "heart_rate_bpm", twin.cursor, "Heart rate", "bpm", "#2F5D9E", lookback),
                   width="stretch", key="chart_3")
    b.plotly_chart(CH.single_series(obs, "hrv_rmssd_ms", twin.cursor, "HRV (RMSSD)", "ms", "#8A6FB0", lookback),
                   width="stretch", key="chart_4")
    a, b = st.columns(2)
    a.plotly_chart(CH.single_series(obs, "steps", twin.cursor, "Steps per 5 min", "steps", "#0F9D8F",
                                    lookback, kind="bar"), width="stretch", key="chart_5")
    sl = twin.observed_sleep()
    if len(sl):
        b.plotly_chart(CH.sleep_bars(sl), width="stretch", key="chart_6")
    else:
        b.info("No completed sleep record yet.")
    meals = twin.observed_meals()
    meals = meals[meals.timestamp >= twin.cursor - pd.Timedelta(hours=lookback)]
    st.markdown("**Self-logged meals in view** (optional input; patients do not log every meal)")
    st.dataframe(meals[["timestamp", "meal_type", "carbs_g_estimated"]].rename(
        columns={"carbs_g_estimated": "carbs (g, patient estimate)"}), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- C fusion
with tabs[2]:
    st.subheader("How the two data streams meet in the twin")
    st.markdown("""
- <span class="tag-ehr">EHR · historical</span> sets the **patient context**: how high this person's
  baseline glucose runs (fasting glucose, HbA1c), and recorded traits linked to how strongly glucose responds
  (BMI, medication class, diabetes duration). It changes slowly — months.
- <span class="tag-sen">Sensors · dynamic</span> update the **current state** every 5 minutes: where glucose is
  now and where it is heading, activity, heart rate/HRV, last night's sleep, optional meal logs.
- **Fusion features** combine both, e.g. *latest CGM minus the patient's own EHR fasting glucose*.
- Every refresh rebuilds all features from the observations received so far and re-runs one trained pipeline.
""", unsafe_allow_html=True)
    f = twin.latest_features.iloc[0]
    rows = [{"Stream": FEATURE_STREAM[k], "Feature": FEATURE_LABELS.get(k, k), "Model input name": k,
             "Current value": (f"{f[k]:.3g}" if isinstance(f[k], float) else str(f[k]))}
            for k in FEATURE_STREAM]
    df_in = pd.DataFrame(rows)
    pick = st.multiselect("Show streams", sorted(df_in.Stream.unique()), default=sorted(df_in.Stream.unique()))
    st.dataframe(df_in[df_in.Stream.isin(pick)], hide_index=True, width="stretch", height=420)
    _, sc = twin.explain()
    st.markdown("#### What each stream adds to this forecast")
    cols = st.columns(len(sc["streams"]) + 1)
    cols[0].markdown(card("Actual forecast", f"{sc['actual']:.0%}", "all inputs as observed"), unsafe_allow_html=True)
    for col, (name, v) in zip(cols[1:], sc["streams"].items()):
        col.markdown(card(f"If {name} were typical", f"{v['prob_if_typical']:.0%}",
                          f"contribution {v['contribution']:+.0%}"), unsafe_allow_html=True)
    st.caption(f"'Typical' = median/mode of the training patients (that patient would score "
               f"{sc['typical_patient_prob']:.0%}). Swapping one stream at a time shows how much the forecast "
               "depends on it. Contributions are not additive and are associations, not causes. "
               "On this synthetic cohort the sensor stream carries most of the signal — see Model evaluation → ablation.")

# --------------------------------------------------------------------------- D prediction
with tabs[3]:
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown(card(f"P(any CGM ≥ {tc['threshold_mg_dl']} mg/dL in next {tc['horizon_min']} min)",
                         f"{fc.probability:.1%}", f'<span class="pill {fc.risk_band}">{fc.risk_band}</span> '
                         f'· forecast made {fc.as_of:%d %b %H:%M}'), unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"- Latest observed glucose: **{fc.latest_cgm:.0f} mg/dL** at {fc.latest_cgm_time:%H:%M}"
                    if fc.latest_cgm else "- No CGM reading available")
        st.markdown(f"- Model alert threshold: **{twin.threshold:.2f}** (chosen on validation patients "
                    f"to reach ≥80% sensitivity) → {'**above**' if fc.above_alert_threshold else 'below'}")
        rb = cfg["risk_bands"]
        st.markdown(f"- Prototype bands: Low < {rb['low_max']:.0%} ≤ Moderate < {rb['moderate_max']:.0%} ≤ "
                    "Elevated. **Not clinically validated**; editable in `config/model_config.yaml`.")
        if fc.warnings:
            for w in fc.warnings:
                st.warning(w)
        else:
            st.success("No data-quality warnings for this forecast.")
    expl, _ = twin.explain()
    with c2:
        st.plotly_chart(CH.contribution_bars(expl), width="stretch", key="chart_7")
    top = expl.head(3)
    up = [r.label for r in top.itertuples() if r.contribution > 0]
    down = [r.label for r in top.itertuples() if r.contribution < 0]
    summary = (f"The model estimates a **{fc.probability:.0%}** chance that glucose reaches "
               f"{tc['threshold_mg_dl']} mg/dL within 2 hours. ")
    if up:
        summary += f"Inputs most associated with a higher estimate here: {', '.join(up)}. "
    if down:
        summary += f"Inputs associated with a lower estimate: {', '.join(down)}. "
    summary += ("This is a statistical estimate on synthetic data for review by a clinician — "
                "it is not a diagnosis or a treatment recommendation.")
    st.info(summary)
    with st.expander("Full explanation table"):
        fmt_v = lambda v: f"{v:.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)  # noqa: E731
        st.dataframe(expl.assign(contribution=expl.contribution.map(lambda v: f"{v:+.3f}"),
                                 value=expl.value.map(fmt_v), reference=expl.reference.map(fmt_v)),
                     hide_index=True, width="stretch")

# --------------------------------------------------------------------------- E simulation
with tabs[4]:
    st.subheader("Digital Twin simulation")
    st.markdown('<span class="tag-sim">SIMULATED</span> Every result below comes from the same trained '
                'model applied to updated inputs. Nothing is typed in as a percentage.', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("##### 1 · Stream the sensor feed forward")
        steps = st.number_input("30-minute steps to replay", 1, 24, 4)
        if st.button("Replay and refresh twin", type="primary", disabled=twin.at_end):
            for _ in range(int(steps)):
                if twin.at_end:
                    break
                twin.advance(30)
            st.rerun()
        st.markdown("##### 2 · Push a manual observation")
        with st.form("manual"):
            mc = st.number_input("CGM reading (mg/dL)", 40, 400, int(fc.latest_cgm or 150))
            mhr = st.number_input("Heart rate (bpm, optional — 0 = skip)", 0, 220, 0)
            go_btn = st.form_submit_button("Ingest at next 5-min tick and refresh")
        if go_btn:
            try:
                twin.ingest_observation(cgm_mg_dl=float(mc), heart_rate_bpm=float(mhr) if mhr else None)
                st.rerun()
            except ValueError as exc:
                st.error(f"Rejected: {exc}")
    with b:
        st.markdown("##### 3 · What-if scenario (twin state is not changed)")
        kind = st.selectbox("Scenario", list(T.SCENARIOS), format_func=lambda k: T.SCENARIOS[k])
        kw = {}
        if kind == "short_sleep":
            kw["sleep_h"] = st.slider("Sleep last night (h)", 3.0, 9.0, 4.5, 0.5)
        if kind == "logged_meal":
            kw["carbs_g"] = st.slider("Carbohydrate (g)", 10, 150, 80, 5)
        try:
            res = twin.scenario(kind, **kw)
            x1, x2, x3 = st.columns(3)
            x1.markdown(card("Baseline forecast", f"{res['baseline'].probability:.0%}", res['baseline'].risk_band),
                        unsafe_allow_html=True)
            x2.markdown(card("Scenario forecast", f"{res['scenario'].probability:.0%}", res['scenario'].risk_band),
                        unsafe_allow_html=True)
            x3.markdown(card("Change", f"{res['delta']:+.0%}", "same model, edited inputs"), unsafe_allow_html=True)
            st.markdown("**Inputs that changed**")
            cf = res["changed_features"].copy()
            cf["feature"] = cf.feature.map(lambda k: FEATURE_LABELS.get(k, k))
            nice = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"  # noqa: E731
            cf["baseline"] = cf["baseline"].map(nice)
            cf["scenario"] = cf["scenario"].map(nice)
            st.dataframe(cf, hide_index=True, width="stretch")
            st.caption("Shows how the model's estimate responds to these inputs. It is NOT a prediction of the "
                       "physiological effect of the action and must not be used as advice.")
        except ValueError as exc:
            st.warning(str(exc))
    st.plotly_chart(CH.risk_history(twin.history_frame(), twin.threshold, cfg["risk_bands"]), width="stretch", key="chart_8")
    with st.expander("Forecast history log"):
        h = twin.history_frame()
        st.dataframe(h[["as_of", "probability", "risk_band", "latest_cgm", "source"]], hide_index=True,
                     width="stretch")

# --------------------------------------------------------------------------- F evaluation
with tabs[5]:
    m = load_metrics()
    if not m:
        st.warning("No evaluation report yet. Run `make evaluate`.")
    else:
        st.subheader("Model evaluation (loaded from reports/metrics.json)")
        st.error(m["disclaimer"])
        lab = m["model_labels"]
        rows = []
        for k, v in m["test"].items():
            ci = m["bootstrap_ci_test"]["models"].get(k, {})
            rows.append({"Model": lab[k], "ROC-AUC": v["roc_auc"], "ROC 95% CI": ci.get("roc_auc"),
                         "PR-AUC": v["pr_auc"], "PR 95% CI": ci.get("pr_auc"), "Brier": v["brier"],
                         "Sensitivity": v["sensitivity"], "Specificity": v["specificity"],
                         "Precision": v["precision"], "F1": v["f1"]})
        st.markdown(f"**Held-out test patients** · {m['splits']['test']['patients']} patients · "
                    f"{m['splits']['test']['rows']:,} prediction points · event prevalence "
                    f"{m['splits']['test']['prevalence']:.1%}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        rows = [{"Model": lab[k], "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"], "Sensitivity": v["sensitivity"],
                 "Specificity": v["specificity"]} for k, v in m["test_onset_subset"].items()]
        st.markdown(f"**Harder subset — glucose not yet ≥ threshold** · {m['splits']['test_onset_subset']['rows']:,} "
                    f"points · prevalence {m['splits']['test_onset_subset']['prevalence']:.1%}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Split methodology:** " + m["splits"]["method"] + ". Temporal check: " +
                    m["temporal_holdout"]["description"] + ".")
        figs = C.FIGURE_DIR
        a, b = st.columns(2)
        for col, name in ((a, "roc_pr_test.png"), (b, "ablation_test.png")):
            if (figs / name).exists():
                col.image(str(figs / name), width="stretch")
        a, b = st.columns(2)
        for col, name in ((a, "calibration_test.png"), (b, "permutation_importance_test.png")):
            if (figs / name).exists():
                col.image(str(figs / name), width="stretch")
        st.markdown("Full report: `reports/metrics.md` · Data provenance: `docs/data_provenance.md` · "
                    "Model card: `docs/model_card.md`")

# --------------------------------------------------------------------------- G responsible use
with tabs[6]:
    st.subheader("Responsible use")
    st.markdown("""
**What this is:** a research and demonstration prototype of a patient digital twin, built and evaluated
entirely on **synthetic** (computer-generated) data.

**What this is not:**
- not a medical device and not cleared or approved by any regulator;
- not validated on real patients — performance numbers measure agreement with our own data generator;
- not a diagnostic tool; it does not make or recommend treatment decisions or medication / insulin dose changes.

**Known limitations:**
- the glucose simulator is a simplified additive model with hand-picked coefficients;
- the 180 mg/dL / 120-minute event rule is a prototype target, not a clinical definition;
- risk bands are illustrative and uncalibrated for clinical use;
- explanations show model associations, not causes; what-if scenarios show model sensitivity, not physiology;
- no live device or hospital-system integration exists; the "stream" is a replay of generated data.

**Privacy:** no real person's data is used. Patient IDs (SYN-xxxx) are fictional; the repository contains
no names, contact details or identifiers.

Any real-world use would require ethics approval, consented real-world data, prospective validation,
clinician oversight and regulatory review. See `docs/responsible_ai.md`.
""")
