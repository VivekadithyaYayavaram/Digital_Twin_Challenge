# Submission readiness audit

Audit date: 2026-10-20. Statuses: **COMPLETE** (verified) · **NEEDS REVIEW** (exists, needs a human check) ·
**PENDING TEAM ACTION** · **BLOCKED** (documented limitation). Re-run `python scripts/check_submission.py`
after any change.

## A. Challenge-required items (README checklist)

| # | Required item | Status | Evidence / action |
|---|---|---|---|
| 1 | Team details | **NEEDS REVIEW** | Team "Innovators", leader Yayavaram Vivekadithya filled (README §1, slide 1). Leader email, members still placeholders. |
| 2 | College / incubator | **NEEDS REVIEW** | Indian Institute of Technology Kharagpur filled; incubator still a placeholder. |
| 3 | Project title | COMPLETE | README §2, slide 1. |
| 4 | Problem statement | COMPLETE | README §3, slide 2. |
| 5 | Healthcare use case | COMPLETE | README §4; target rule in `config/model_config.yaml`. |
| 6 | Technical stack | COMPLETE | README §9; `requirements.txt`. |
| 7 | AI/ML model + framework | COMPLETE | README §11, `docs/model_card.md`, `artifacts/model_metadata.json`. |
| 8 | 20-minute demo video link | **PENDING TEAM ACTION** | Script ready: `demo/DEMO_SCRIPT.md`. Record, upload **Unlisted**, paste URL in README (top table + §17). No URL was fabricated. |
| 9 | Open-source license | COMPLETE | `LICENSE` (MIT), README §19. Copyright holder "SwasthTwin contributors" — team may rename. |
| 10 | Architecture diagram PDF/PPT | COMPLETE | `submission/architecture_diagram.{pdf,pptx}`; PPTX passed OOXML validation; PDF rendered and visually checked. |
| 11 | Presentation PDF/PPT | **NEEDS REVIEW** | `submission/Innovators_IIT_Kharagpur.{pdf,pptx}` — 10 slides, speaker notes, validated, every page rendered and checked. Named `Innovators_IIT_Kharagpur` per TeamName_Institution rule; slide 1 shows team, institution, leader. |
| 12 | Public access without permissions | **PENDING TEAM ACTION** | Repo not yet pushed; nothing was published. Push to GitHub, set **Public**, open every link in an incognito window. Not verified. |

**Video length discrepancy:** the brief says both "minimum 20 minutes" and "15–20 minutes". The official rules
page (unstop.com listing) renders client-side and could not be read during the build. **NEEDS REVIEW** by the team.

## B. Technical dimensions (judge's view)

| Dimension | Status | Evidence |
|---|---|---|
| Problem clarity | COMPLETE | Single target, horizon and threshold defined in config; README §3–4. |
| Use-case relevance | COMPLETE | T2D / prediabetes, CGM + EHR; non-prescriptive decision support. |
| Real static + dynamic fusion | COMPLETE | EHR parameters drive the simulator (`data_generator._gen_ehr` → carb sensitivity); model uses 13 EHR + 24 sensor + 2 fusion + 3 context inputs; ablation in `reports/metrics.md`. Honest finding: EHR adds little beyond sensors on this data. |
| Working Digital Twin | COMPLETE | `PatientTwin` state, replay, manual ingest, scenarios, history; tests `test_twin_*` (state updates, prediction == direct model call, scenarios do not mutate state). |
| Leakage prevention | COMPLETE | `test_features_identical_when_future_is_removed`, `test_future_changes_do_not_affect_features`, `test_meal_logged_after_t_not_visible`, `test_label_matches_future_window_only`, `test_patient_level_split_is_disjoint`; runtime assert in `step_evaluate`. |
| Reproducible evaluation | COMPLETE | Fresh clone + new venv + `make clean && make pipeline` reproduced `reports/metrics.json` and `data/ehr_patients.csv` **byte-for-byte**; `test_evaluation_is_reproducible`. |
| Metrics honesty | COMPLETE | All figures generated from `reports/metrics.json`; `check_submission.py` confirms README, model card and slides match. Weaknesses stated (sensitivity 0.772 < 0.80 target; time-of-day shortcut; synthetic only). |
| Dashboard usability | COMPLETE | 7 tabs; headless `AppTest` (start, +30 min, replay, 4 scenarios, patient switch) and Playwright screenshots in `submission/screenshots/`. Fixed during audit: stale sidebar clock, duplicate chart IDs, clipped banner and labels, Arrow type warning. |
| Graceful failure | COMPLETE | Missing data/model → "Setup needed" + one-click build (tested on a clean clone); invalid manual readings rejected; stale/sparse sensor warnings. |
| Privacy & responsible AI | COMPLETE | Synthetic only, fictional IDs, no PII columns (tested), no secrets (scanned), disclaimer on every view, `docs/responsible_ai.md`. |
| Documentation | COMPLETE | README (20 sections), architecture, data dictionary, provenance, model card, design system, demo script. All relative links resolve (scripted check). |
| Architecture diagram quality | COMPLETE | Separates training-time and runtime lanes, marks the replay feed as simulated, shows artifact hand-off and refresh loop. |
| Presentation quality | NEEDS REVIEW | Visual QA done; team should rehearse and personalise slide 1. |
| Repository accessibility | PENDING TEAM ACTION | Local git repo with one commit; not pushed. |

## C. Commands verified in this build

| Command | Result |
|---|---|
| `pip install -r requirements.txt` (fresh venv, Python 3.13) | passed |
| `make data` / `make train` / `make evaluate` | passed (≈2 min total) |
| `make test` (30 tests) | **30 passed** (main repo and fresh clone) |
| `make app` / `streamlit run app.py` | served HTTP 200; screenshots captured |
| `make submission` (`build_architecture.py`, `build_presentation.py`) | passed; needs LibreOffice + poppler (`pdftoppm`) |
| `python scripts/check_submission.py` | 0 errors; 3 PENDING (team info, video URL, repo URL) |
| Windows / macOS | **not tested** — only Linux was available |

## D. Known limitations (not bugs)

* Synthetic, hand-tuned simulator; results do not transfer to real patients.
* Model leans on time of day because synthetic meals follow routines.
* Test sensitivity 0.772 vs 0.80 validation target.
* 16 test patients → wide CIs; sensor-gap subset only 366 rows (no CI).
* No repeat-seed (multiple cohorts) variability study — only patient bootstrap CIs.
* No calibration step (probabilities near 0/1); bands uncalibrated.
* Screenshot capture needs Playwright + Chromium (optional dev tool, not in `requirements.txt`).

## E. Team to-do before submitting

1. Add incubator, leader email and members in `config/team_info.yaml` and README §1.
2. Confirm the official video length rule.
3. Record the demo (presenter visible) from `demo/DEMO_SCRIPT.md`; upload as **Unlisted**.
4. Paste the video URL and public repo URL into README; replace `<PUBLIC_REPO_URL>`.
5. Push to GitHub, make public, test all links in incognito.
6. `make test && python scripts/check_submission.py` on a fresh clone.
