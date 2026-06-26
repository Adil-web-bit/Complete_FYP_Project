# Deployment Notes

## Local run

```bash
cd human_behavior_predictor
python -m venv .venv
```

Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

- This app writes data to local files:
  - `data/raw/monitoring_sessions.jsonl`
  - `models/behavior_model.joblib`
  - `models/training_report.json`
  - `models/evaluation_report.json`
- On Streamlit Community Cloud, the filesystem may be **ephemeral**. Your JSONL sessions and model artifacts may not
  persist reliably across restarts/deployments.

## Production recommendations

- Replace local JSONL storage with a proper database/object storage.
- Add authentication and access control.
- Record explicit consent (who/when/what) and retention policies.
- Encrypt and secure sensitive data.
- Validate model performance carefully; do not claim medical/psychological accuracy.

## Public datasets

- Do not commit public datasets unless the dataset license explicitly allows redistribution.
- `data/public/` is gitignored by default.
- Verify dataset licensing and usage terms before use.

## Monitoring component note

- The monitoring component currently loads the Streamlit component library from a CDN (`unpkg.com`).
- If the CDN is unavailable, the app provides a manual JSON paste fallback in the Monitoring page expander.
