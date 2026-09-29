# Industrial Machine Predictive Maintenance — Dashboard

A production-style Streamlit deployment for the stacking-ensemble predictive
maintenance model (Random Forest + Decision Tree + KNN + Logistic Regression
→ Logistic Regression meta-model).

## What's included

```
deployment/
├── app/
│   ├── app.py                     # Streamlit dashboard (UI + inference)
│   ├── pipeline_utils.py          # IQRCapper transformer (shared with training)
│   ├── stack_model.joblib         # Trained stacking ensemble
│   ├── base_models.joblib         # Trained base models (RF, DT, KNN, LR)
│   ├── model_report.json          # Test-set metrics, CV results, ROC data
│   ├── requirements.txt
│   └── sample_batch_template.csv  # Example file for the batch-scoring tab
├── training/
│   ├── train.py                   # Full retraining pipeline (your original script, cleaned up)
│   └── pipeline_utils.py
├── .streamlit/config.toml         # Dark theme
├── Dockerfile
└── README.md
```

## What the dashboard adds beyond the original script

- **Live prediction tab** — sidebar form instead of blocking `input()` calls; instant
  results with a color-coded risk banner, health score, failure probability, and
  model-agreement metrics.
- **Interactive Plotly graphs** replacing the static matplotlib figures: probability
  bar chart, base-model comparison, and a radar/spider chart of machine condition —
  all zoomable/hoverable instead of static PNGs.
- **Batch scoring tab** — upload a CSV of many machines, score them all at once,
  see a distribution histogram, and download the results.
- **Model performance tab** — confusion matrix, ROC curve, and per-model CV
  comparison computed once at training time and stored in `model_report.json`,
  so the dashboard doesn't need the raw training data or GridSearchCV at runtime.
- **Dataset overview tab** — class balance and a summary of the feature
  engineering / leakage-control choices, for transparency with stakeholders.
- **Condition analysis + maintenance recommendations** — same threshold logic as
  your script, shown as readable cards instead of console text.
- Model and report are loaded once via `st.cache_resource`, so the app stays fast.

## Run locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy for free — Streamlit Community Cloud

1. Push this `deployment/` folder to a GitHub repo (the `app/` folder must contain
   `app.py`, `requirements.txt`, and the three model artifact files).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick the repo, set the main file path to `app/app.py`.
4. Deploy. You'll get a public `*.streamlit.app` URL in a couple of minutes.

## Deploy with Docker (any cloud VM, Render, Railway, AWS/GCP/Azure)

```bash
cd deployment
docker build -t predictive-maintenance-app .
docker run -p 8501:8501 predictive-maintenance-app
```

Push the image to a registry (Docker Hub / ECR / GCR) to deploy on any container
platform — Render, Railway, Fly.io, ECS, Cloud Run, or an Azure Web App for
Containers all accept this image as-is.

## Retraining on new data

1. Put the updated CSV in `training/` (or edit `DATA_PATH` in `training/train.py`).
2. `cd training && pip install -r ../app/requirements.txt && python train.py`
3. Copy the three regenerated files (`stack_model.joblib`, `base_models.joblib`,
   `model_report.json`) into `app/`, replacing the old ones.
4. Redeploy (or just restart the app — `st.cache_resource` will pick up new files
   only on a fresh process, so a restart is required after swapping the artifacts).

## Things worth adding next (not included here)

- **Authentication** — Streamlit Community Cloud supports viewer allow-lists; for
  self-hosting, put the app behind an auth proxy (e.g. Cloudflare Access, OAuth2 Proxy)
  before exposing it externally, since it currently has no login.
- **Drift monitoring** — log incoming predictions (e.g. to a small SQLite/Postgres
  table) and periodically compare live feature distributions to the training set.
- **CI for retraining** — a scheduled GitHub Action that reruns `train.py` on fresh
  data and opens a PR with updated artifacts, rather than manual copying.
- **Model versioning** — if this grows past one model, consider MLflow or a simple
  version suffix on the `.joblib` filenames plus a "model version" field in
  `model_report.json`.
- **Unit tests** — a small pytest suite asserting `engineer_features` output and
  that `stack.predict` returns 0/1, to catch schema drift early.

## Model summary (from the last training run)

| Metric | Value |
|---|---|
| Accuracy | 97.1% |
| Precision | 94.6% |
| Recall | 96.6% |
| F1 | 95.6% |
| ROC-AUC | 0.996 |
| PR-AUC | 0.992 |

Full details, including per-base-model CV scores and hyperparameters, are in
`app/model_report.json` and viewable in the dashboard's **Model Performance** tab.
