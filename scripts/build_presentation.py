"""Build submission/Innovators_IIT_Kharagpur.pptx (+ .pdf), 10 slides with speaker notes.

Every number on the slides is read from reports/metrics.json or computed by
running the saved model through the Digital Twin at build time. Nothing is
typed in by hand. Run after `make pipeline` (and the screenshot script).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from pptx_helpers import (AMBER, AMBER_LIGHT, CARD, EHR, EHR_LIGHT, INK, LINE, MUTED, NAVY, NAVY2, RED,  # noqa: E402
                          SEN, SEN_LIGHT, TEAL, TEAL_LIGHT, WHITE, arrow, blank, box, bullets, circle_num,
                          label_box, new_deck, text, to_pdf)

from swasthtwin import config as C  # noqa: E402
from swasthtwin import data_generator as G  # noqa: E402
from swasthtwin import twin as T  # noqa: E402

OUT = ROOT / "submission"
SHOTS = OUT / "screenshots"
M = 0.6
TW = 13.333 - 2 * M


def pct(v, nd=0):
    return "n/a" if v is None else f"{v * 100:.{nd}f}%"


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def title(s, t, sub=None, dark=False):
    text(s, M, 0.42, TW, 0.7, [[(t, {"bold": True, "size": 30, "color": WHITE if dark else NAVY})]])
    if sub:
        text(s, M, 1.12, TW, 0.45, sub, size=15, color=RGB_SUB_DARK if dark else MUTED)


RGB_SUB_DARK = TEAL_LIGHT


def footer(s, n, dark=False):
    text(s, M, 7.05, 9, 0.28, "SwasthTwin · research prototype on synthetic data · not a medical device",
         size=10, color=TEAL_LIGHT if dark else MUTED)
    text(s, 12.2, 7.05, 0.55, 0.28, str(n), size=10, color=TEAL_LIGHT if dark else MUTED, align=PP_ALIGN.RIGHT)


def team_value(v: str | None, fallback: str) -> tuple[str, object]:
    if not v or str(v).startswith(("TODO", "PENDING")):
        return f"[{fallback} — to be added]", AMBER
    return str(v), WHITE


def bar_chart(s, x, y, w, h, cats, series: dict[str, list[float]], colors, title_txt, vmin=0.0, vmax=1.0,
              fmt='0.000', legend=True):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = title_txt
    tp = ch.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size = Pt(13); tp.runs[0].font.bold = True; tp.runs[0].font.color.rgb = NAVY
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = INK
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = vmin, vmax
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = MUTED
    va.format.line.fill.background()
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.color.rgb = INK
    ca.format.line.color.rgb = LINE
    ca.reverse_order = True
    plot = ch.plots[0]
    plot.gap_width = 60; plot.overlap = -10
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(10); dl.font.color.rgb = INK
    for ser, col in zip(plot.series, colors):
        if isinstance(col, list):
            for i, c in enumerate(col):
                pt = ser.points[i]
                pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
        else:
            ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
    return gf


def stat(s, x, y, w, big, small, color=NAVY, fill=CARD):
    box(s, x, y, w, 1.35, fill=fill)
    text(s, x + 0.2, y + 0.12, w - 0.4, 0.7, [[(big, {"bold": True, "size": 32, "color": color})]])
    text(s, x + 0.2, y + 0.8, w - 0.4, 0.5, small, size=12, color=MUTED)


def example_scenario(meta, pipe):
    """Run the real twin on a held-out test patient to get honest example numbers."""
    cohort = G.load_cohort()
    split = pd.read_csv(C.ARTIFACT_DIR / C.SPLIT_FILE)
    pid = split[split.split == "test"].patient_id.iloc[0]
    tw = T.PatientTwin(pid, cohort, pipe, meta)
    first = tw.refresh()
    for _ in range(12):  # +6 h, same as the screenshot script (3 x '+2 h')
        tw.advance(30)
    walk = tw.scenario("walk_30min")
    meal = tw.scenario("logged_meal", carbs_g=80)
    hist = tw.history_frame()
    return {"pid": pid, "ehr": tw.ehr, "first": first, "now": tw.latest, "walk": walk, "meal": meal,
            "min_p": float(hist.probability.min()), "max_p": float(hist.probability.max()),
            "n": len(hist)}


def crop(src: Path, box_px, name: str) -> Path:
    from PIL import Image
    im = Image.open(src)
    out = OUT / "screenshots" / name
    im.crop(box_px).save(out)
    return out


def build() -> Path:
    m = json.loads((C.REPORT_DIR / C.METRICS_FILE).read_text())
    team = C.load_team_info()
    pipe, meta = T.load_model()
    ex = example_scenario(meta, pipe)
    best = m["selected_model"]
    lab = m["model_labels"]
    tb, tr_, te = m["test"][best], m["test"]["rule_baseline"], m["test"]["logistic_regression"]
    on_b, on_r = m["test_onset_subset"][best], m["test_onset_subset"]["rule_baseline"]
    ci = m["bootstrap_ci_test"]["models"]
    ab = m["ablation"]
    sp = m["splits"]
    thr = m["target"]["threshold_mg_dl"]; hz = m["target"]["horizon_min"]

    prs = new_deck()

    # 1 ── title ───────────────────────────────────────────────────────────────
    s = blank(prs, NAVY)
    text(s, M, 1.55, 11, 0.5, "HAPPIEST HEALTH DIGITAL TWIN CHALLENGE 2026", size=14, color=TEAL_LIGHT, bold=True)
    text(s, M, 2.05, 12, 1.0, [[("SwasthTwin", {"bold": True, "size": 54, "color": WHITE})]])
    text(s, M, 3.1, 11.5, 1.0, "A personalised Digital Twin that forecasts the risk of a hyperglycemia event "
         "in the next two hours by fusing historical EHR data with wearable and CGM streams", size=20, color=WHITE)
    tn, tc = team_value(team.get("team_name"), "Team name")
    inst, ic = team_value(team.get("institution"), "Institution")
    lead, lc = team_value((team.get("team_leader") or {}).get("name"), "Team leader")
    text(s, M, 4.75, 11, 1.3, [[("Team  ", {"size": 15, "color": TEAL_LIGHT}), (tn, {"size": 15, "color": tc, "bold": True})],
                               [("Institution  ", {"size": 15, "color": TEAL_LIGHT}), (inst, {"size": 15, "color": ic})],
                               [("Team leader  ", {"size": 15, "color": TEAL_LIGHT}), (lead, {"size": 15, "color": lc})]],
         line_spacing=1.2)
    box(s, M, 6.25, 6.2, 0.5, fill=NAVY2, radius=0.1)
    text(s, M + 0.15, 6.32, 6.0, 0.4, "Research prototype · synthetic data only · not a medical device",
         size=13, color=AMBER_LIGHT, bold=True)
    s.notes_slide.notes_text_frame.text = (
        "Introduce the team and the one-line idea: a per-patient digital twin that combines a person's medical "
        "history with live wearable data to estimate the chance of a high-glucose event in the next two hours. "
        "Say up front that everything shown runs on synthetic data and is a research prototype.")

    # 2 ── problem ──────────────────────────────────────────────────────────────
    s = blank(prs)
    title(s, "The problem: high-glucose episodes are seen after they happen",
          "Adults with or at risk of type 2 diabetes generate rich CGM and wearable data, but it is reviewed "
          "retrospectively")
    cards = [("1", "Data without foresight", "CGM shows where glucose IS. Reports summarise where it WAS. "
              "Few tools estimate where it is heading for this specific person."),
             ("2", "Context is split", "The EHR (HbA1c, diagnosis, medicines) and the wearable stream "
              "(glucose, activity, sleep) sit in different systems and are rarely read together."),
             ("3", "Clinician time is short", "A care team cannot watch every stream. They need a short, "
              "explained signal: who may be heading high soon, and why the model thinks so.")]
    for i, (n, h, b) in enumerate(cards):
        x = M + i * (TW / 3 + 0.0)
        cw = TW / 3 - 0.3
        box(s, x, 1.95, cw, 2.9, fill=CARD)
        circle_num(s, x + 0.3, 2.2, 0.55, n)
        text(s, x + 0.3, 2.95, cw - 0.6, 0.5, [[(h, {"bold": True, "size": 19, "color": NAVY})]])
        text(s, x + 0.3, 3.5, cw - 0.6, 1.3, b, size=14, color=INK)
    box(s, M, 5.15, TW - 0.3, 1.55, fill=TEAL_LIGHT)
    text(s, M + 0.3, 5.3, TW - 0.9, 0.4, [[("Specific use case", {"bold": True, "size": 16, "color": TEAL})]])
    text(s, M + 0.3, 5.72, TW - 0.9, 0.9,
         f"Every 30 minutes, estimate P(any CGM reading ≥ {thr} mg/dL in the next {hz} minutes) for one patient, "
         "explain the estimate and flag data-quality problems — for clinician review, not for automatic action.",
         size=15, color=INK)
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "Why this use case: hyperglycemia excursions are frequent in type 2 diabetes and CGM data is already "
        "collected, but it is mostly reviewed after the fact. A short-horizon, explained warning is a concrete, "
        f"testable task. The {thr} mg/dL / {hz}-minute rule is our prototype target, not a clinical definition. "
        "We avoid quoting prevalence statistics we have not verified.")

    # 3 ── twin concept + workflow ───────────────────────────────────────────────
    s = blank(prs)
    title(s, "Why it is a Digital Twin, not just a prediction model",
          "Each patient has a persistent virtual state that is updated as new observations arrive")
    steps = [("Observe", "EHR profile + new sensor readings up to the twin's clock"),
             ("Update state", "Rebuild rolling features with the same code used in training"),
             ("Forecast", "Saved model scores the 120-minute event risk"),
             ("Explain", "Top inputs, stream contributions, data-quality warnings"),
             ("Review", "Clinician sees trend + history; tries what-if inputs")]
    sw = (TW - 4 * 0.25) / 5
    for i, (h, b) in enumerate(steps):
        x = M + i * (sw + 0.25)
        box(s, x, 1.95, sw, 2.0, fill=CARD)
        circle_num(s, x + 0.2, 2.15, 0.5, str(i + 1))
        text(s, x + 0.2, 2.75, sw - 0.4, 0.4, [[(h, {"bold": True, "size": 17, "color": NAVY})]])
        text(s, x + 0.2, 3.15, sw - 0.35, 0.8, b, size=13, color=INK)
        if i < 4:
            arrow(s, x + sw + 0.02, 2.95, x + sw + 0.23, 2.95, color=TEAL, width=2)
    text(s, M, 4.3, 6, 0.4, [[("What makes it a twin", {"bold": True, "size": 18, "color": NAVY})]])
    bullets(s, M, 4.8, 6.0, 2.1, [
        "Patient-specific state object (EHR + observations + features + forecast history)",
        "Advances in time: deterministic replay of the sensor stream, or manual readings",
        "Every refresh is a real model call on updated inputs — no scripted percentages",
        "What-if mode edits a copy of the inputs and re-scores; the twin itself is unchanged"], size=14)
    box(s, 7.0, 4.3, TW - 6.4, 2.5, fill=AMBER_LIGHT)
    text(s, 7.25, 4.45, TW - 6.9, 0.4, [[("Honest scope", {"bold": True, "size": 16, "color": AMBER})]])
    bullets(s, 7.25, 4.9, TW - 6.9, 1.9, [
        "Data-driven twin, not a mechanistic physiology model",
        "No live wearable or hospital integration — the feed is a replay",
        "What-if shows model sensitivity, not the body's response"], size=14)
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "A plain classifier scores rows. The twin keeps state per patient and is updated over time: in the demo we "
        "advance the clock, the twin ingests the next part of the stream, recomputes features and calls the model "
        "again, and keeps a history. Be clear about scope: this is a data-driven twin; there is no live device "
        "integration; and what-if results describe how the model responds, not physiology.")

    # 4 ── two-stream fusion ─────────────────────────────────────────────────────
    s = blank(prs)
    title(s, "Two-stream fusion: sensors carry most signal, the EHR adds context",
          "Same algorithm trained on different inputs, evaluated on held-out test patients")
    label_box(s, M, 1.95, 3.2, 2.35, "EHR · historical", "Age, sex, BMI, diabetes status and duration, "
              "HbA1c, fasting glucose, eGFR, lipids, BP, medication class", EHR_LIGHT, EHR, line=EHR,
              title_size=16, body_size=13)
    label_box(s, M, 4.45, 3.2, 2.35, "Sensors · dynamic", "CGM level, trend, variability, coverage; heart "
              "rate, HRV vs own 24 h; steps; last night's sleep; optional meal log", SEN_LIGHT, SEN, line=SEN,
              title_size=16, body_size=13)
    label_box(s, M + 3.4, 3.2, 2.1, 2.35, "Fusion features", "Latest CGM minus the patient's EHR fasting "
              "glucose; HbA1c × logged carbs", TEAL_LIGHT, TEAL, line=TEAL, title_size=15, body_size=12.5)
    arrow(s, M + 3.22, 3.1, M + 3.38, 3.7, color=EHR, width=2)
    arrow(s, M + 3.22, 5.6, M + 3.38, 5.0, color=SEN, width=2)
    names = ["EHR + time of day", "Sensor + time of day", "Fused (all inputs)"]
    keys = ["ehr_only", "sensor_only", "fused_ehr_plus_sensor"]
    bar_chart(s, 6.3, 1.85, 6.4, 3.45, names,
              {"All test rows": [ab[k]["test"]["pr_auc"] for k in keys],
               "Glucose not yet high": [ab[k]["test_onset_subset"]["pr_auc"] for k in keys],
               "During CGM gaps": [ab[k]["test_sensor_gap_subset"]["pr_auc"] for k in keys]},
              [TEAL, SEN, AMBER], "PR-AUC by input set (higher is better)", vmin=0.5, vmax=1.0)
    gain_all = ab["fused_ehr_plus_sensor"]["test"]["pr_auc"] - ab["sensor_only"]["test"]["pr_auc"]
    gain_gap = (ab["fused_ehr_plus_sensor"]["test_sensor_gap_subset"]["pr_auc"]
                - ab["sensor_only"]["test_sensor_gap_subset"]["pr_auc"])
    abci = m["ablation_bootstrap_ci"]["models"]
    bullets(s, 6.3, 5.45, 6.4, 1.5, [
        f"Adding the EHR to sensors: +{gain_all:.3f} PR-AUC overall, +{gain_gap:.3f} during CGM gaps.",
        f"Small gain; confidence intervals overlap (fused {f3(abci['fused_ehr_plus_sensor']['pr_auc'][0])}–"
        f"{f3(abci['fused_ehr_plus_sensor']['pr_auc'][1])} vs sensor {f3(abci['sensor_only']['pr_auc'][0])}–"
        f"{f3(abci['sensor_only']['pr_auc'][1])}). We do not claim fusion is proven to help."], size=13, space=5)
    footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        "Left: what each stream contributes. Right: an ablation run by the evaluation script, same algorithm, "
        "different inputs, both single-stream models also get time of day. Honest message: on this synthetic "
        "cohort the sensor stream carries most of the signal; adding the EHR helps a little, most during CGM gaps, "
        "and the intervals overlap. Real data could show a different picture; that is a question for validation.")

    # 5 ── architecture ───────────────────────────────────────────────────────────
    s = blank(prs)
    title(s, "System architecture: offline training, runtime twin")
    arch_png = OUT / "architecture_diagram.png"
    subprocess.run(["pdftoppm", "-png", "-r", "160", "-singlefile", str(OUT / "architecture_diagram.pdf"),
                    str(arch_png.with_suffix(""))], check=True)
    ah = 5.65
    aw = ah * 13.333 / 7.5
    s.shapes.add_picture(str(arch_png), Inches((13.333 - aw) / 2), Inches(1.2), height=Inches(ah))
    footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        "Top lane: offline. Generate linked synthetic data, validate, build features using only past data and "
        "labels from the next two hours, split by patient, train and select on validation patients, save the "
        "pipeline, evaluate on test patients. Bottom lane: runtime. The twin rebuilds features with the same "
        "function, the saved model scores them, the dashboard shows risk, explanation and warnings. The dashed "
        "box is the honest bit: the sensor feed is a replay, not a live device.")

    # 6 ── data, features, privacy ─────────────────────────────────────────────────
    s = blank(prs)
    title(s, "Synthetic data, leakage-safe features, privacy by design",
          f"{sp['train']['patients'] + sp['val']['patients'] + sp['test']['patients']} fictional patients × "
          f"14 days of 5-minute data, fully reproducible from one seed")
    cols = [("Synthetic cohort", TEAL, [
        "EHR drives the simulator: higher HbA1c / BMI → larger meal spikes; meds damp them",
        "Meals, walks, stress, short sleep and dawn rise shape glucose; HR/HRV respond",
        "Realistic mess: CGM gaps, watch-off hours, noisy and incomplete meal logs",
        "Hand-picked coefficients — documented, not fitted to real data"]),
        ("Features (all from the past)", SEN, [
         "CGM: level, 15/30-min trend, mean / SD / min / max, time in range, coverage",
         "Wearable: HR, HRV vs own 24 h, steps, active minutes; last night's sleep",
         "Meals count only from the moment they are logged",
         "Test: features identical whether or not future data exists"]),
        ("Privacy & safety", EHR, [
         "No real person's data; IDs are SYN-0001…",
         "No names, phone numbers or government IDs anywhere",
         "Runs fully offline; no API keys or cloud services",
         "Prominent not-a-medical-device notice in every view"])]
    cw = (TW - 2 * 0.3) / 3
    for i, (h, col, items) in enumerate(cols):
        x = M + i * (cw + 0.3)
        box(s, x, 1.9, cw, 4.1, fill=CARD)
        text(s, x + 0.25, 2.05, cw - 0.5, 0.45, [[(h, {"bold": True, "size": 18, "color": col})]])
        bullets(s, x + 0.25, 2.65, cw - 0.5, 4.0, items, size=14, space=9)
    footer(s, 6)
    s.notes_slide.notes_text_frame.text = (
        "The generator is deterministic. The EHR is not decoration: it sets each patient's baseline and how strongly "
        "they respond to meals, so the two streams are genuinely linked. We added realistic problems: sensor gaps, "
        "watch-off periods, and patients who log only some meals with wrong carb estimates. All features use only "
        "past data and an automated test proves it. No real data and no identifiers are used.")

    # 7 ── ML approach ────────────────────────────────────────────────────────────
    s = blank(prs)
    title(s, "ML approach and leakage-resistant evaluation",
          "Simple baselines first; choose the model on validation patients; touch test patients once")
    models = [("Rule baseline", "No ML: alert if current or 60-min trend-projected CGM ≥ 180"),
              ("Logistic regression", "Linear baseline ML model"),
              ("Random forest", "Nonlinear ensemble"),
              ("Gradient boosting", "Histogram gradient-boosted trees")]
    for i, (h, b) in enumerate(models):
        y = 1.95 + i * 0.95
        sel = (i == 3 and best == "hist_gradient_boosting") or (i == 2 and best == "random_forest") or \
              (i == 1 and best == "logistic_regression")
        box(s, M, y, 5.6, 0.8, fill=TEAL_LIGHT if sel else CARD, line=TEAL if sel else None)
        text(s, M + 0.25, y + 0.1, 5.2, 0.65, [[(h + ("   (selected)" if sel else ""),
                                                  {"bold": True, "size": 15, "color": TEAL if sel else NAVY})],
                                                [(b, {"size": 12.5, "color": MUTED})]])
    text(s, M, 5.85, 5.6, 0.9, "scikit-learn pipelines: median imputation + missing-value indicators, "
         "one-hot encoding, model. Saved with joblib with feature list, threshold and version.", size=12.5, color=MUTED)
    x2 = M + 6.0
    text(s, x2, 1.9, 6, 0.4, [[("Evaluation design", {"bold": True, "size": 18, "color": NAVY})]])
    bullets(s, x2, 2.4, TW - 6.0, 4.4, [
        [("Split by patient: ", {"bold": True}), (f"{sp['train']['patients']} train / {sp['val']['patients']} "
          f"validation / {sp['test']['patients']} test patients — nobody appears in two splits", {})],
        [("Temporal check: ", {"bold": True}), ("train on days 0–9 of training patients, test on days 10+ of "
          "test patients", {})],
        [("Selection: ", {"bold": True}), ("highest validation PR-AUC; alert threshold set on validation for ≥80% "
          "sensitivity", {})],
        [("Uncertainty: ", {"bold": True}), ("95% CIs from a patient-level bootstrap "
          f"({m['bootstrap_ci_test']['method'].split(',')[1].strip()})", {})],
        [("Harder subset: ", {"bold": True}), ("rows where glucose is not yet ≥ 180 — persistence alone cannot "
          "score well here", {})],
        [("Also reported: ", {"bold": True}), ("Brier score, calibration curve, confusion matrix, prevalence, "
          "permutation importance", {})]], size=14, space=8)
    footer(s, 7)
    s.notes_slide.notes_text_frame.text = (
        "We compare against a non-ML rule and a linear model before trusting anything complex. Model choice uses "
        "validation patients only. The split is by patient, so the test measures performance on people the model "
        "has never seen; a second check also moves forward in time. We report the subset where glucose is not "
        "already high because there, simply saying 'it is high now' does not work.")

    # 8 ── results ────────────────────────────────────────────────────────────────
    s = blank(prs)
    title(s, "Results on held-out synthetic patients",
          f"{sp['test']['patients']} test patients · {sp['test']['rows']:,} forecasts · event prevalence "
          f"{pct(sp['test']['prevalence'])} · all numbers from reports/metrics.json")
    order = ["rule_baseline", "logistic_regression", "random_forest", best]
    order = list(dict.fromkeys(order))
    short = {"rule_baseline": "Rule baseline", "logistic_regression": "Logistic regression",
             "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting"}
    bar_chart(s, M, 1.8, 6.6, 3.6, [short[k] for k in order],
              {"All test rows": [m["test"][k]["pr_auc"] for k in order],
               "Glucose not yet high": [m["test_onset_subset"][k]["pr_auc"] for k in order]},
              [TEAL, SEN], "PR-AUC on test patients", vmin=0.0, vmax=1.0)
    x2 = M + 6.9
    sw2 = (TW - 6.9 - 0.25) / 2
    stat(s, x2, 1.85, sw2, f3(tb["pr_auc"]), f"{short[best]} PR-AUC\n95% CI {f3(ci[best]['pr_auc'][0])}–"
         f"{f3(ci[best]['pr_auc'][1])}", color=TEAL)
    stat(s, x2 + sw2 + 0.25, 1.85, sw2, f3(tr_["pr_auc"]), f"Rule baseline PR-AUC\n95% CI "
         f"{f3(ci['rule_baseline']['pr_auc'][0])}–{f3(ci['rule_baseline']['pr_auc'][1])}", color=MUTED)
    stat(s, x2, 3.4, sw2, pct(tb["sensitivity"]), f"sensitivity at threshold {tb['threshold']:.2f}\n"
         f"(target was 80% on validation)", color=NAVY)
    stat(s, x2 + sw2 + 0.25, 3.4, sw2, pct(tb["specificity"]), f"specificity · precision {pct(tb['precision'])}",
         color=NAVY)
    box(s, M, 5.6, TW, 1.2, fill=AMBER_LIGHT)
    tm = m["temporal_holdout"][best]
    text(s, M + 0.25, 5.68, TW - 0.5, 1.1, [
        [("What this does and does not show  ", {"bold": True, "size": 14, "color": AMBER})],
        [(f"Where glucose is not yet high, PR-AUC {f3(on_b['pr_auc'])} vs {f3(on_r['pr_auc'])} for the rule. "
          f"Temporal hold-out PR-AUC {f3(tm['pr_auc'])}. Test sensitivity {pct(tb['sensitivity'])} fell short of the "
          "80% validation target. Scores are high partly because the generator is learnable (e.g. regular meal "
          "times make clock time predictive). This is agreement with our simulator, NOT clinical evidence.",
          {"size": 13, "color": INK})]])
    footer(s, 8)
    s.notes_slide.notes_text_frame.text = (
        f"Read the numbers carefully. Gradient boosting reaches PR-AUC {f3(tb['pr_auc'])} on test patients versus "
        f"{f3(tr_['pr_auc'])} for the no-ML rule and {f3(te['pr_auc'])} for logistic regression. The more meaningful "
        f"comparison is the subset where glucose is not already high: {f3(on_b['pr_auc'])} versus {f3(on_r['pr_auc'])}. "
        f"Sensitivity on test was {pct(tb['sensitivity'])}, below the 80% we targeted on validation, which shows that "
        "thresholds do not transfer perfectly. The biggest caveat: high scores reflect how learnable our own "
        "simulator is; time of day is a top feature because synthetic meals follow routines. None of this is "
        "evidence about real patients.")

    # 9 ── live prototype ────────────────────────────────────────────────────────
    s = blank(prs)
    title(s, "Live prototype: one synthetic patient, step by step",
          f"Held-out patient {ex['pid']} · {ex['ehr']['diabetes_status'].replace('_', ' ')} · HbA1c "
          f"{ex['ehr']['hba1c_pct']}% · values below produced by the saved model at build time")
    tl = crop(SHOTS / "02_timeline.png", (560, 520, 2290, 1110), "slide_timeline.png")
    sim = crop(SHOTS / "05_simulation.png", (1430, 600, 2290, 1360), "slide_whatif.png")
    s.shapes.add_picture(str(tl), Inches(M), Inches(1.85), width=Inches(8.1))
    s.shapes.add_picture(str(sim), Inches(M + 8.4), Inches(1.75), height=Inches(3.15))
    now, walk, meal = ex["now"], ex["walk"], ex["meal"]
    bullets(s, M, 5.1, TW, 1.8, [
        f"Twin replayed {ex['n'] - 1} half-hour updates; its risk estimate ranged {pct(ex['min_p'])}–{pct(ex['max_p'])} "
        "as the stream changed — every value a fresh model call.",
        f"At {now.as_of:%a %H:%M}: latest CGM {now.latest_cgm:.0f} mg/dL, P(event in 2 h) {pct(now.probability)} "
        f"({now.risk_band}). The explanation shows time of day as a top driver — a known limitation of the synthetic routine.",
        f"What-if (model sensitivity, not advice): +30-min walk → {pct(walk['scenario'].probability)}; "
        f"logging 80 g carbs now → {pct(meal['scenario'].probability)}."], size=14, space=6)
    footer(s, 9)
    s.notes_slide.notes_text_frame.text = (
        "Walk through the dashboard as in the demo video: pick a held-out patient, advance the clock, watch the risk "
        "history update, open the explanation, then run what-if scenarios. Point out that the shaded forecast "
        "window contains no invented glucose values, and that the explanation honestly shows time of day as a "
        "top driver, which is a limitation of synthetic routines. What-if results show the model's sensitivity, "
        "not what would happen to the patient.")

    # 10 ── value, limitations, roadmap ────────────────────────────────────────────
    s = blank(prs, NAVY)
    title(s, "Potential value, limitations and the road to validation", dark=True)
    text(s, M, 1.3, 4.0, 0.4, [[("Potential value (unproven)", {"bold": True, "size": 17, "color": TEAL_LIGHT})]])
    bullets(s, M, 1.8, 3.9, 3.0, [
        "Earlier, explained flag for clinician review",
        "One place to read EHR + wearable context",
        "Data-quality warnings instead of silent guesses",
        "Reusable twin pattern for other forecasts"], size=14, color=WHITE)
    text(s, M + 4.25, 1.3, 4.0, 0.4, [[("Key limitations", {"bold": True, "size": 17, "color": AMBER_LIGHT})]])
    bullets(s, M + 4.25, 1.8, 3.9, 3.6, [
        "Synthetic, hand-tuned simulator; results do not transfer",
        "180 mg/dL / 2 h rule is a prototype target",
        "Clock time over-predictive due to regular synthetic meals",
        "Risk bands uncalibrated for clinical use",
        "No device or EHR integration; no hypoglycemia target"], size=14, color=WHITE)
    text(s, M + 8.5, 1.3, 3.8, 0.4, [[("Roadmap", {"bold": True, "size": 17, "color": TEAL_LIGHT})]])
    road = [("1", "Public de-identified CGM data (licence checked) — retrain, re-test"),
            ("2", "Clinicians co-define target, threshold and alert burden"),
            ("3", "Retrospective study with ethics approval"),
            ("4", "Silent prospective pilot; calibration and safety review"),
            ("5", "Regulatory pathway before any clinical use")]
    for i, (n, t) in enumerate(road):
        y = 1.85 + i * 0.85
        circle_num(s, M + 8.5, y, 0.42, n, size=12)
        text(s, M + 9.05, y - 0.02, 3.2, 0.8, t, size=13, color=WHITE)
    box(s, M, 6.05, TW, 0.75, fill=NAVY2, radius=0.1)
    text(s, M + 0.25, 6.15, TW - 0.5, 0.6, "Ask: feedback from clinicians on the target definition and alert "
         "threshold, and access routes to consented real-world CGM + EHR data for validation.", size=14,
         color=WHITE, bold=True)
    footer(s, 10, dark=True)
    s.notes_slide.notes_text_frame.text = (
        "Close honestly. The value is potential, not demonstrated. Name the limitations plainly: synthetic data, a "
        "prototype target, clock-time shortcut, uncalibrated bands, no live integration. The roadmap moves from "
        "public de-identified data, to clinician-defined targets, to ethics-approved retrospective and then silent "
        "prospective evaluation, and only then regulatory review.")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "Innovators_IIT_Kharagpur.pptx"
    prs.save(p)
    pdf = to_pdf(p)
    print("saved", p, pdf)
    return p


if __name__ == "__main__":
    build()
