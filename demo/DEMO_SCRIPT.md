# SwasthTwin — 20-minute demonstration script

**Target length: ~20:00.** The challenge text mentions both a *minimum 20-minute* demo and a *15–20 minute*
video. We could not read the official rules page while building this. **Check the official submission
instructions and confirm the allowed length before recording.** This script is timed to land at 19:30–20:00,
which satisfies "20 minutes" without going over a 20-minute cap. If the rules say "at least 20 minutes",
extend Section 8 by 1–2 minutes (step through a second patient).

**Presenter on camera:** record picture-in-picture (webcam bubble bottom-right) for the whole video, or at least
for sections 1, 2, 10 and 11. See "Recording setup".

Everything shown must be live in the running app. Numbers quoted below are from the committed run
(seed 2026); if you regenerate data, read the new numbers from `reports/metrics.md` and the dashboard instead.

---

## Before you press record

```bash
pip install -r requirements.txt
make test            # expect: all tests pass
make app             # opens http://localhost:8501
```

Open in separate tabs: the dashboard, `submission/Innovators_IIT_Kharagpur.pdf`,
`submission/architecture_diagram.pdf`, `reports/metrics.md`, and the GitHub repo page.
Click **Reset this patient's twin** so the clock starts at **Wed 04 Mar 2026, 07:00**.
Keep the default patient (first held-out test patient, **SYN-0002**).

---

## Timeline

| # | Time | Section | Screen |
|---|---|---|---|
| 1 | 00:00–01:30 | Introduction and problem | Slides 1–2 |
| 2 | 01:30–03:00 | Why this use case | Slide 2 |
| 3 | 03:00–05:00 | Digital Twin concept and architecture | Slides 3, 5 / architecture PDF |
| 4 | 05:00–06:30 | Static and dynamic data sources | Slide 4, dashboard **Two-stream fusion** |
| 5 | 06:30–08:00 | Data generation and privacy | Slide 6, `docs/data_provenance.md` |
| 6 | 08:00–10:00 | Feature engineering and model design | Slide 7, `src/swasthtwin/features.py` |
| 7 | 10:00–12:00 | Evaluation metrics and what they mean | Slide 8, dashboard **Model evaluation** |
| 8 | 12:00–16:30 | Live dashboard demo | Dashboard tabs |
| 9 | 16:30–18:00 | Simulated patient-state update + refreshed prediction | **Digital Twin simulation** |
| 10 | 18:00–19:15 | Limitations, safety, roadmap | Slide 10, **Responsible use** |
| 11 | 19:15–20:00 | Summary | Slide 1 or camera |

---

## 1 · Introduction and problem (00:00–01:30) — camera + slide 1

> "Hello, we are [TEAM NAME] from [INSTITUTION]. This is SwasthTwin, a personalised digital twin that estimates
> the chance of a high-glucose event in the next two hours for an adult with type 2 diabetes or prediabetes.
> Everything you will see runs on synthetic, computer-generated patients. It is a research prototype, not a
> medical device."

Slide 2: "People wearing CGMs produce glucose readings every five minutes, but those readings are mostly
reviewed after the fact. The medical record and the wearable stream sit in different places, and a care team
cannot watch every stream."

## 2 · Why this use case (01:30–03:00) — slide 2

> "We chose one narrow, testable question: every 30 minutes, will any CGM reading reach 180 mg/dL in the next
> 120 minutes? It is short enough to act on, it uses data that CGM users already produce, and it can be
> scored objectively. The 180 / 120 rule is our prototype target — not a clinical definition — and it is
> configurable in `config/model_config.yaml`."

## 3 · Digital Twin concept and architecture (03:00–05:00) — slides 3 and 5

> "A prediction model scores a row. A digital twin keeps a virtual state per patient and updates it over time."

Walk the five steps on slide 3: observe → update state → forecast → explain → review.
Open the architecture slide. Top lane: offline training. Bottom lane: runtime twin. Point to:
* the **same feature function** in both lanes (no train/serve skew);
* the **dashed sensor-feed box**: "this is a replay of generated data — we have not integrated any device or hospital system";
* the feedback loop: advancing the clock, entering a reading or running a what-if makes the twin recompute.

## 4 · Static and dynamic data (05:00–06:30) — slide 4, then dashboard

Slide 4 left side: EHR stream vs sensor stream vs fusion features.
Switch to dashboard → **Two-stream fusion** tab. Scroll the input table, filter to "EHR (historical)", then
"Sensor (dynamic)". Point at the "If … were typical" cards:

> "Here we swap one whole stream for a typical patient's values and re-score. For this patient the sensor
> stream moves the estimate far more than the EHR does."

Back to slide 4 chart: "The ablation agrees: sensors carry most of the signal; adding the EHR gives a small
gain, largest during CGM gaps, and the confidence intervals overlap — so we do not claim fusion is proven."

## 5 · Data generation and privacy (06:30–08:00) — slide 6

> "No real patient data is used. A seeded generator creates 80 fictional patients with 14 days of 5-minute data.
> The EHR is linked to the physiology: higher HbA1c and BMI give larger meal spikes, and medication damps them.
> We added realistic mess: sensor gaps, watch-off hours, and meal logs that are incomplete and wrong by about
> 20 percent. The coefficients are hand-picked and documented in `docs/data_provenance.md`. IDs are SYN-numbers;
> there are no names or contact details anywhere."

## 6 · Feature engineering and model design (08:00–10:00) — slide 7

> "Features use trailing windows only: current glucose, 15- and 30-minute trend, variability, time in range,
> heart rate, HRV against the person's own last 24 hours, steps, last night's sleep, logged carbs, time of day,
> and the EHR fields. Meals count only from the moment they are logged."

Show `tests/test_features_leakage.py::test_features_identical_when_future_is_removed` briefly:
"This test computes features on the full stream and on a stream cut at time t, and requires them to be identical."

Models: "We start with a no-ML rule — alert if glucose is high now or trending there — then logistic regression,
random forest and gradient boosting. We choose on validation patients by PR-AUC and set the alert threshold on
validation patients for 80 % sensitivity."

## 7 · Evaluation (10:00–12:00) — slide 8, dashboard **Model evaluation**

> "The split is by patient — 48 train, 16 validation, 16 test — so the test is on people the model never saw.
> On those 16 synthetic patients gradient boosting reaches PR-AUC 0.975, versus 0.880 for the rule and 0.943 for
> logistic regression. PR-AUC is the area under the precision–recall curve: how well alerts concentrate on real
> events. The more meaningful number is where glucose is not already high: 0.867 versus 0.472 for the rule."

Be explicit about the weaknesses:
> "Test sensitivity was 77 %, below the 80 % we set on validation — thresholds do not transfer perfectly.
> Scores are high partly because our generator is learnable; time of day is a top feature because synthetic
> meals follow routines. These numbers measure agreement with our own simulator, not clinical accuracy."

Show the calibration and ablation images on the evaluation tab.

## 8 · Live dashboard (12:00–16:30)

1. **Overview** (12:00): read the probability card, latest CGM, alert threshold, twin update time, EHR table
   (purple tag) and current twin state (blue tag). "Risk bands are prototype bands, not validated."
2. Sidebar → **⏩ +2 h** three times (12:45). Watch the clock move 07:00 → 13:00 and the risk-history chart grow.
   "Each click replays the next part of the stream, recomputes features and calls the saved model."
3. **Patient timeline** (13:30): solid line = observed CGM; shaded area = the window being forecast. "We never
   draw predicted glucose values." Toggle **Reveal what happened next** — "only possible in a replay, hidden from
   the model". Show heart rate, HRV, steps, sleep bars, logged meals.
4. **Prediction & explanation** (14:45): read the probability, the top inputs chart, the plain-language
   summary. "Time of day is a top driver here — this is the routine-learning limitation I mentioned."
   Open **Full explanation table**.
5. Switch patient (15:45): turn off "Held-out test patients only" if needed, pick a prediabetes patient, compare
   the EHR table and the lower baseline risk.

## 9 · Simulated state update and refreshed prediction (16:30–18:00) — **Digital Twin simulation**

1. Set "30-minute steps to replay" to 2, click **Replay and refresh twin** — the risk history gains points.
2. In **Push a manual observation**, enter CGM **240**, click **Ingest at next 5-min tick and refresh**. The
   overview card and history update; source shows "manual".
3. In **What-if**, choose the walk scenario: show baseline vs scenario and the changed inputs table. Then
   "Log a carbohydrate-rich meal" at 80 g; then "short sleep" at 4.5 h.
   > "These are the same trained model on edited inputs. They show how the model responds, not what would
   > happen in the body, and they are not advice."
4. Click **Reset this patient's twin** to show state is per-patient.

## 10 · Limitations, safety, roadmap (18:00–19:15) — slide 10, **Responsible use**

Limitations: synthetic simulator; prototype target; time-of-day shortcut; uncalibrated bands; no device/EHR
integration; no hypoglycemia; small test cohort. Roadmap: public de-identified CGM data (licence verified) →
clinician-defined target and alert burden → ethics-approved retrospective study → silent prospective pilot →
regulatory pathway.

## 11 · Summary (19:15–20:00) — camera

> "SwasthTwin keeps a per-patient state that fuses medical history with live sensor data, refreshes a two-hour
> risk estimate with a real model on every update, explains it, and is honest about what it cannot yet claim.
> The code, data generator, tests, slides and this script are in our public repository. Thank you."

---

## Recording setup

* **Tool:** OBS Studio (free) — Scene = Display Capture + Video Capture Device (webcam) resized to ~20 % in a
  corner. Or Zoom: share screen with camera on, record locally.
* 1920×1080, 30 fps, MP4 (H.264). Browser zoom 90 % so dashboard cards fit. Close notifications.
* Use a headset microphone; record a 10-second test and play it back.
* Rehearse once with a timer; adjust section 8 to hit the target length.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Setup needed" on the dashboard | run `make data && make train && make evaluate`, or click the in-app button |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Clock does not move | the replay reached the end of 14 days — click **Reset this patient's twin** |
| Evaluation tab says no report | run `make evaluate` |
| Numbers differ from this script | you regenerated data with a different seed/config; quote `reports/metrics.md` |
| Slow first load | first load reads ~3 MB of CSV and the model; wait a few seconds |

## Upload checklist

- [ ] Length confirmed against the official rules (≈20:00, see note at top)
- [ ] Presenter visible on camera; project visible on screen
- [ ] Audio clear throughout; no private notifications or personal data on screen
- [ ] Upload to YouTube as **Unlisted** (not Private — private videos need sign-in)
- [ ] Open the link in a private/incognito window to confirm it plays without signing in
- [ ] Paste the URL into `README.md` (Deliverables table) and `config/team_info.yaml` → `links.demo_video`
- [ ] Update `SUBMISSION_READINESS.md` item "Demo video" to COMPLETE
