# ICU Alert Engine

Event-driven alert engine for **sustained tachycardia** and **hypoxaemia** in the intensive care unit, with silent composite physiological indices used to stratify alert severity.

This repository accompanies the manuscript *"Development and Retrospective Characterisation of an Event-Driven Alert Engine with New Composite Physiological Indices for Sustained Tachycardia and Hypoxaemia in the Intensive Care Unit: A COVID-19-Motivated Study".

> **Research prototype, not for clinical use.** The engine has been characterised retrospectively only. Its alerts and severity labels have not been validated against clinical outcomes or expert annotation.

---

## How the engine works

| Component | Rule (default) |
|---|---|
| Primary alert: tachycardia | Heart rate > 100 bpm in **N = 2** consecutive valid observations |
| Primary alert: hypoxaemia | SpO₂ < 90% in **N = 2** consecutive valid observations |
| Gap reset | Breach state reset (no resolution event) if valid observations are > 900 s apart |
| Plausibility bounds | HR 20–250 bpm, SpO₂ 50–100%, SBP 30–250 mmHg, RR 4–60 breaths/min |
| Resolution event | First valid observation back within the normal range after an alert |

Each alert is classified as **critical** if at least one associated confirmer is in sustained breach when the alert fires (OR rule), and as **high** otherwise. Confirmers never emit alerts of their own.

| Confirmer | Definition | Breach | Confirms |
|---|---|---|---|
| Shock Index | HR / SBP | > 1.0 | Tachycardia |
| HR/SpO₂ ratio | HR / SpO₂ | > 1.10 | Tachycardia, hypoxaemia |
| SBP/SpO₂ ratio | SBP / SpO₂ | < 1.0 | Tachycardia |
| RR/SpO₂ ratio | RR / SpO₂ | > 0.30 | Hypoxaemia |
| SBP × SpO₂ product | SBP × SpO₂ | < 9,000 | Hypoxaemia |

All thresholds are set in `AlertConfig` (see `icu_alert_engine.py`). The three indices beyond the Shock Index and RR/SpO₂ are exploratory and are not validated clinical scores.

---

## Repository structure

```
.
├── icu_alert_engine.py      # Alert engine (state machine, confirmers, AlertConfig, CLI)
├── preparar_dados.py        # Data preparation: invalid timestamps, duplicates, missing values
├── demo.py                  # Prepares DATA_FINAL.csv and runs the engine (main execution)
├── analise_revisao.py       # Complementary analyses reported in the manuscript
├── verificar_exemplos.py    # Checks the four example alerts quoted in the manuscript
├── sample_vitals.csv        # Small synthetic file for testing without the clinical dataset
├── requirements.txt
└── resultados/              # Outputs of analise_revisao.py (aggregate values only)
```

---

## Requirements

- Python 3.12 (tested with 3.12.10)
- pandas 2.3.3
- numpy

```bash
pip install -r requirements.txt
```

---

## Data

The clinical dataset (`DATA_FINAL.csv`, adult cardiac ICU, Hospital de Santa Maria, 2008–2025) **is not included** and must never be committed to this repository. Access is subject to institutional approval.

The scripts expect the file **one level above** the repository folder:

```
Tese/
├── DATA_FINAL.csv
└── Sistema_detecao/        # this repository
```

Alternatively, pass the path as an argument (see below).

Required columns (other columns are ignored):

| Column | Content |
|---|---|
| `PATIENT_ID` | Patient identifier (read as text) |
| `ADMISSION_ID` | ICU admission identifier (optional, used for per-admission statistics) |
| `RTDATA_TIME` | Observation timestamp |
| `FREQ_CARDIACA` | Heart rate (bpm) |
| `SPO2` | Peripheral oxygen saturation (%) |
| `PRESSAO_ARTERIAL_SISTOLICA` | Systolic blood pressure (mmHg) |
| `FREQ_RESPIRATORIA` | Respiratory rate (breaths/min) |

---

## Usage

### Quick test with synthetic data

```bash
python3 icu_alert_engine.py sample_vitals.csv
python3 icu_alert_engine.py sample_vitals.csv --json     # full alert payloads
```

### Reproducing the results in the manuscript

```bash
# 1. (optional) write the prepared dataset to ../DATA_FINAL_LIMPO.csv and print the exclusion counts
python3 preparar_dados.py

# 2. run the engine on the prepared data and print the alert summary
python3 demo.py

# 3. complementary analyses: writes the results to resultados/
python3 analise_revisao.py                  # uses ../DATA_FINAL.csv
python3 analise_revisao.py path/to/file.csv # or an explicit path

# 4. check the four example alerts quoted in the manuscript
python3 verificar_exemplos.py
```

`analise_revisao.py` runs the engine once over the full dataset and may take several minutes.

### Data preparation steps

`preparar_dados.py` applies three steps in a fixed order and reports what each one removed:

1. Records with an unparseable timestamp or outside 1 January 2008 to 31 December 2025.
2. Duplicate records with the same `PATIENT_ID` and `RTDATA_TIME`; the first occurrence in file order is kept.
3. Records with a missing value in any of HR, SpO₂, SBP or RR (complete-case analysis).

Values outside the plausibility bounds are **not** removed here; the engine ignores them observation by observation.

### Using the engine in code

```python
from icu_alert_engine import AlertEngine, AlertConfig

engine = AlertEngine(AlertConfig(persistence_samples=2, max_gap_seconds=900),
                     on_alert=lambda ev: print(ev.to_push_payload()))
events = engine.process_csv("sample_vitals.csv")
```

`process_sample()` is available for streaming one observation at a time, and `process_dataframe()` for a pandas DataFrame.

---

## Outputs of `analise_revisao.py`

All files are written to `resultados/`.

| File | Content |
|---|---|
| `analise_revisao.json` | All results in raw form |
| `results_macros.tex` | All numerical values cited in the manuscript |
| `tab_flow.tex` | Data-preparation flow (records, patients, admissions per step) |
| `tab_quality.tex` | Missingness and plausibility-bound exclusions in the source file |
| `tab_events.tex`, `tab_config.tex` | Alerts by condition; one versus three confirmers per condition |
| `tab_burden.tex` | Alerts per patient and per admission |
| `tab_episodes.tex` | Alert–resolution matching, gap resets, episode duration |
| `tab_ablation.tex`, `tab_corr.tex` | Contribution of each confirmer; Spearman correlations |
| `tab_danger.tex` | Effect of absolute danger thresholds on severity |
| `tab_temporal.tex` | Data characteristics and alert behaviour by calendar period |

All outputs contain aggregate values only and no patient identifiers.

---

## Reproducibility

The results reported in the manuscript were obtained with Python 3.12.10 and pandas 2.3.3 on an Apple M2 workstation running macOS Tahoe 26.5, using the default `AlertConfig`.

## Ethics

The study is a retrospective secondary analysis of anonymised, routinely collected monitoring data. The ethics approval reference will be added once available.
