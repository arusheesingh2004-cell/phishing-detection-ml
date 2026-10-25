# Phishing URL Detection Using Machine Learning

A web-based phishing URL detection project with a FastAPI backend, a browser-based frontend, URL feature extraction, and trained Logistic Regression and XGBoost models.

> **Research/demo tool only.** Predictions are probabilistic and may be wrong. Do not use this project as the only protection against phishing or open suspicious URLs to test them. A URL classified as legitimate is not guaranteed to be safe.

## Features

- Classifies a URL as **phishing** or **legitimate** using trained ML models.
- Offers Logistic Regression and XGBoost model options.
- Extracts URL-based features locally, such as URL length, subdomain count, suspicious terms, digit ratio, and URL structure.
- Provides single-URL and batch prediction API endpoints.
- Supports uploading a `.csv` or `.txt` file for batch predictions.
- Stores recent prediction history in a local SQLite database by default.
- Displays model evaluation metrics through the web interface/API.

## Technology stack

- Python
- FastAPI and Uvicorn
- Pandas, NumPy, scikit-learn, XGBoost, and Joblib
- SQLAlchemy with SQLite by default
- HTML, CSS, and JavaScript frontend

## Repository structure

```text
phishing-detection-ml/
├── backend/
│   ├── app.py
│   ├── feature_extractor.py
│   ├── train_model.py
│   └── models/
│       ├── best_model.txt
│       ├── logistic_regression.joblib
│       ├── scaler.joblib
│       ├── xgboost.joblib
│       └── metrics.json
├── data/
│   └── raw_dataset.csv
├── frontend/
│   └── index.html
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or 3.11 recommended.
- `pip` installed.

## Run locally

### 1. Download the repository

Clone it with Git:

```bash
git clone https://github.com/YOUR-USERNAME/phishing-detection-ml.git
cd phishing-detection-ml
```

Or download the repository ZIP from GitHub and extract it.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

From the repository root:

```bash
cd backend
python app.py
```

Open **http://127.0.0.1:8000** in your browser. Interactive API documentation is available at **http://127.0.0.1:8000/docs**.

The backend serves the frontend from `frontend/index.html`. The default database is created locally as `backend/phishing_detector.db` when the application starts; it is not intended to be committed.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check application/model status |
| `GET` | `/api/model-info` | View saved model metrics |
| `POST` | `/api/predict` | Predict for one URL |
| `POST` | `/api/predict/batch` | Predict for a list of URLs |
| `POST` | `/api/predict/batch-file` | Predict URLs from a CSV/TXT upload |
| `GET` | `/api/history` | View recent prediction history |

For request formats, open `/docs` after starting the server.

## Model evaluation

The saved metrics file reports the following held-out test-set results for the included model artifacts. These are the project's recorded results, not a guarantee of performance on new or real-world URLs.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 80.75% | 87.88% |
| Precision | 84.03% | 89.87% |
| Recall | 75.94% | 85.39% |
| F1-score | 79.78% | 87.57% |
| ROC-AUC | 88.66% | 95.22% |

Evaluation uses a train/test split recorded in `backend/models/metrics.json`. Results can vary by dataset, split, and deployment environment. The model should be evaluated on additional, independently collected data before making security claims.

## Retrain the models

The included training script reads `data/raw_dataset.csv` by default when run from the `backend` directory:

```bash
cd backend
python train_model.py
```

Check `python train_model.py --help` for supported options. Training may overwrite model artifacts and metrics, so back up the current files if you want to retain them.

## Dataset note

The included CSV is the dataset supplied with this project. Before making this repository public, confirm and document the dataset's original source, license, and redistribution permissions. Add a citation or source link here when verified.

## Limitations and responsible use

- URL features alone cannot reliably determine whether a website is malicious in every case.
- Threats and attacker techniques change over time; retraining and independent evaluation are needed.
- False positives and false negatives are possible.
- Do not enter private URLs or sensitive tokens into a public deployment.
- This project is for educational and research purposes and is not a substitute for browser, email, endpoint, or network security controls.

## Future improvements

- Add independent time-based and domain-separated evaluation.
- Add unit tests and automated checks.
- Document the dataset provenance and license.
- Deploy with HTTPS, rate limiting, and production-safe database configuration.

## Author

Add your name and relevant links here before publishing.
