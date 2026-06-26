# Human Behavior Predictor (Scaffold)

A modular Streamlit app scaffold for predicting human behavior from typing and scrolling behavior.

This repository intentionally contains **placeholders** for:
- collecting typing/scrolling data
- feature extraction
- model training
- prediction

## Folder structure

```
human_behavior_predictor/
  app.py
  requirements.txt
  README.md
  .gitignore
  src/
  data/
  tests/
```

## Quickstart (exact commands)

```powershell
cd human_behavior_predictor
```

### 1) Create virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
```

**macOS/Linux**
```bash
python3 -m venv .venv
```

### 2) Install requirements

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run Streamlit app

```bash
streamlit run app.py
```

### 4) Run tests

```bash
pytest
```

## Next development step

Simplified demo workflow:
1. Run the app.
2. Give consent.
3. Click **Start Monitoring** inside the widget.
4. Type and scroll inside the widget.
5. Click **Stop Monitoring** inside the widget.
6. Click **Predict Mood**.

Model behavior:
- If `models/behavior_model.joblib` exists, predictions use the trained ML model.
- Otherwise, predictions fall back to a rule-based baseline.

Developer note:
- Training, evaluation, public dataset utilities, export/import, and local JSONL storage modules still exist in `src/`,
  but the Streamlit UI is demo-focused by default.

## Advanced Training Page

The app includes a developer-only **Training** page to train or replace the saved model. The **Home** page is for
prediction only and will automatically use the latest saved model.

Recommended order for EmoSurv CSVs:
1. Frequency Dataset
2. Fixed Text Typing Dataset
3. Free Text Typing Dataset

Do not use **Participants Information Dataset** alone for mood prediction (it is demographic, not typing behavior).

Notes:
- Every training run overwrites `models/behavior_model.joblib`.
- Write down the reported validation accuracy before trying the next CSV.

Privacy reminder:
- Data is stored locally on the machine running the app.
- Only interactions inside the monitoring component are captured (not system-wide keystrokes).
- Do not use this to secretly monitor anyone; always get explicit consent.

Deployment notes: see `DEPLOYMENT.md`.

## Manual Entry (Optional)

Earlier scaffold steps supported manual entry for estimated typing/scrolling signals. The current preferred path is the
**Real Monitoring MVP** below, which captures raw-ish events via browser JavaScript.

## Real Monitoring MVP

This version includes a practical monitoring demo using browser JavaScript inside Streamlit (no React/npm build).

Exact flow:
1. Run the app.
2. Open the **Home** page.
3. Enter `user_id` (or keep `anonymous`).
4. Give consent.
5. In the embedded monitoring widget, click **Start Monitoring**.
7. Type naturally in the typing box.
8. Scroll inside the scroll area.
9. In the widget, click **Stop Monitoring**.
10. Click **Predict Mood**.

Notes:
- Captured events are returned directly to Python using a lightweight Streamlit custom component.
- A manual copy/paste fallback is available in the Home page expander if automatic transfer fails.
- Predictions are baseline/rule-based until enough labeled sessions exist and a trained model is added.
- The Home page shows a clean prediction result and compact feature tables; raw debug JSON is hidden under **Technical details**.

EmoSurv model compatibility note:
- Live sessions now capture both `keydown` and `keyup` and extract EmoSurv-style timing features (e.g., `D1U1`, `D1D2`),
  improving compatibility with models trained on EmoSurv Fixed/Free Text datasets.

## Automatic Browser Event Capture

The app uses a Streamlit custom component (static HTML/JS/CSS) under `components/monitoring_component/` to capture:
- typing `keydown` + `keyup` events
- scroll events inside the component scroll box

Captured data is returned to Streamlit/Python via the official component protocol (`Streamlit.setComponentValue(...)`).

Component version: `0.4.0`

Reliability notes:
- The component bridge is loaded by trying `components/monitoring_component/frontend/vendor/streamlit-component-lib.js` first, then falling back to a CDN (`unpkg.com`).
- If the bridge fails to load/return data, use the Home page expander: **Advanced fallback: paste captured JSON manually** (emergency only).
- When you stop monitoring, the latest stopped payload is kept in-memory so you can click **Predict Mood**.
- Use **Reset** to clear the latest payload and prediction.

Local run instructions (no npm build step required):
- `cd human_behavior_predictor`
- `streamlit run app.py`

Troubleshooting (blank widget):
- If the monitoring widget area is blank, the Streamlit component bridge may not have loaded.
- Ensure `components/monitoring_component/frontend/index.html`, `main.js`, and `style.css` exist.
- If you are offline or `unpkg.com` is blocked, vendor the bridge file to `components/monitoring_component/frontend/vendor/streamlit-component-lib.js`.
- Check the browser DevTools console for script loading errors.
- Use the **Advanced fallback: paste captured JSON manually** expander only as an emergency workaround.

Consent and privacy:
- Data is stored locally on the machine running the app.
- The app only captures interactions inside the monitoring component (not system-wide keystrokes).
- Do not use this to secretly monitor anyone; always get explicit consent.

## Dataset and Training Plan

- Public keystroke datasets can help validate typing feature extraction.
- Most public keystroke datasets do not directly provide mood labels like `focused`/`stressed`/`rushed`/`distracted`.
- For this app, the best MVP strategy is to collect **self-labeled** monitoring sessions from users (with consent).
- Once you have ~100–300 labeled sessions, train a supervised classifier and persist it to `models/behavior_model.joblib`.
- External datasets can be added later for pretraining or benchmarking, but they are not used in this step.

This app is a behavioral pattern demo and does not claim medical accuracy.

## Training a Real Model

Developer-only (not exposed in the simplified UI by default):
1. Collect labeled sessions (requires extending the UI or using backend utilities).
2. Minimum to train: **10 labeled sessions** (100–300+ recommended).
3. Train using backend modules (e.g., `src/model/trainer.py`), which writes `models/behavior_model.joblib`.
4. The Home page will automatically use `models/behavior_model.joblib` when available; otherwise it uses rule-based fallback.

## Training from a Public Typing Dataset

- Recommended first dataset: **EmoSurv** (emotion-labeled keystroke dynamics).
- EmoSurv-style labels are mapped to app labels:
  - Neutral -> `normal`
  - Calmness -> `focused`
  - Anger -> `stressed`
  - Happiness -> `focused`
  - Sadness -> `distracted`
- EmoSurv may also use `EmotionIndex` / `Emotion Index` with short codes:
  - `H` (Happy) -> `focused`
  - `S` (Sad) -> `distracted`
  - `A` (Angry) -> `stressed`
  - `C` (Calm) -> `focused`
  - `N` (Neutral) -> `normal`
- KeyRecs and CMU keystroke datasets are primarily authentication/typing rhythm datasets (often not mood-labeled). If you
  use them, treat results as pipeline testing only and do not claim mood accuracy.

How to use:
1. Put a CSV at `data/public/public_keystroke_dataset.csv` (this folder is gitignored).
2. The CSV must include a label/emotion column (e.g., `emotion`, `Emotion`, `label`, `mood`) and numeric typing feature columns.
3. Train using backend modules (e.g., `src/model/public_dataset_trainer.py`), or re-enable the training UI if needed.

Before training, use the built-in **Dataset Schema Preview**:
- confirm the detected label column
- confirm mapped label counts (and how many rows were dropped/unmapped)
- confirm the final numeric feature count
- if many rows are dropped, label mapping may not match your dataset labels
- if no numeric features are detected, your CSV may contain raw events rather than extracted features
- troubleshooting: if preview shows rows but `cols=1`, the file delimiter is likely not comma. The loader auto-detects
  common delimiters (comma/semicolon/tab/pipe) and encodings, but if issues remain, re-save as **CSV UTF-8 comma-delimited**.

Public typing datasets are typing-only (no scrolling) and may not match browser-captured app data. A public-trained model is
a quick demo baseline, not proof of real-world mood prediction.

### EmoSurv (recommended first file)

EmoSurv includes multiple CSVs (names may vary):
- Fixed Text Typing Dataset
- Free Text Typing Dataset
- Frequency Dataset
- Participants Information Dataset

Recommended first file:
- **Frequency Dataset**

Why:
- It typically already contains task-level numeric features and an `EmotionIndex` label, so it trains quickly.

Steps:
1. Download the EmoSurv CSV files.
2. Rename the Frequency Dataset CSV to `public_keystroke_dataset.csv`.
3. Put it at `data/public/public_keystroke_dataset.csv`.
4. App → **Training → Public Typing Dataset Training**.
5. Click **Preview Local CSV**.
6. If preview status is `ok`, click **Train from Local Public CSV**.

Notes:
- Fixed/free typing datasets can also work if they include numeric timing features.
- Participants Information should not be used alone for mood prediction (demographics, not typing behavior).

EmoSurv Fixed/Free formatting note:
- The loader attempts to convert numeric-looking timing columns (e.g., `D1U1`, `U1D2`, `key Down`, `key Up`) from strings
  like `"20,5"` or `" 4.2 "` into real numeric features.
- If schema preview still shows very few features, inspect the CSV formatting for mixed text values in timing columns.

Limitations:
- Self-reported labels can be noisy.
- Small datasets produce unreliable models.
- This is not a medical or psychological diagnosis.
- For production, collect more data and validate performance carefully.

## Exporting and Importing Sessions

Use the **Saved Sessions** page:
- **Export Sessions** writes a `.jsonl` file under `data/exports/`.
- **Anonymize export** (recommended) replaces `user_id` with `"anonymous"` and blanks `typed_text`.
  Raw event timing remains because it is needed for feature extraction.
- **Import** accepts a local `.jsonl` file and deduplicates by `session_id`.
- **Backup Sessions** creates a timestamped backup in `data/exports/`.
- **Danger Zone** can clear the local sessions file (models are not deleted).

All export/import is local only; nothing is uploaded anywhere.

## Model Evaluation

After training, the app writes:
- `models/training_report.json`
- `models/evaluation_report.json`

The **Evaluation** page displays validation accuracy (if available), label distribution, classification report, and a
confusion matrix. Small datasets may skip validation and accuracy on tiny datasets is not reliable.
