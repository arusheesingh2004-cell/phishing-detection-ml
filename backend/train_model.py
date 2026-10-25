"""
train_model.py

Trains and compares Logistic Regression and XGBoost on lexical URL features,
and picks whichever gives the best balance of detection performance and
computational efficiency -- the project's second objective.

Works across dataset sizes:
  - Any CSV with a URL column and a label column (names configurable --
    not hard-wired to one dataset's schema).
  - Reads in chunks so a file far bigger than the demo dataset doesn't need
    to fit in memory all at once; --sample-size caps memory further via
    reservoir sampling, so even a file bigger than RAM can be used.
  - Small datasets (a few hundred rows or fewer) automatically switch from a
    single train/test split to stratified k-fold cross-validation, because a
    single small holdout set gives a noisy, unreliable accuracy estimate.

Examples:
    # default demo dataset
    python train_model.py

    # a different dataset with different column names
    python train_model.py --data-path my_urls.csv --url-col link --label-col class --phishing-label 1

    # cap a huge file to a random 200k-row sample while training
    python train_model.py --data-path huge_file.csv --sample-size 200000 --n-jobs 4
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from xgboost import XGBClassifier

from feature_extractor import extract_features, FEATURE_NAMES, extract_features_matrix

RANDOM_STATE = 42
SMALL_DATASET_THRESHOLD = 500  # below this, prefer cross-validation over a single split
DEFAULT_CHUNKSIZE = 50_000


def parse_args():
    p = argparse.ArgumentParser(description="Train phishing URL detection models.")
    p.add_argument("--data-path", default="../data/raw_dataset.csv",
                    help="Path to a CSV with a URL column and a label column.")
    p.add_argument("--url-col", default=None,
                    help="Name of the URL column. Auto-detected if omitted (tries: url, URL, link, Link).")
    p.add_argument("--label-col", default=None,
                    help="Name of the label column. Auto-detected if omitted (tries: status, label, Label, class, Result).")
    p.add_argument("--phishing-label", default=None,
                    help="Value in --label-col that means 'phishing' (e.g. 'phishing', '1', '-1'). "
                         "Auto-detected for common encodings if omitted.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--sample-size", type=int, default=None,
                    help="If the dataset has more rows than this, take a uniform random sample of this size "
                         "via reservoir sampling while reading (keeps memory flat regardless of file size).")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                    help="Rows read per chunk from the CSV. Lower this for very memory-constrained machines.")
    p.add_argument("--n-jobs", type=int, default=1,
                    help="Worker processes for feature extraction on large datasets.")
    p.add_argument("--cv-folds", type=int, default=0,
                    help="Force k-fold cross-validation with this many folds regardless of dataset size. "
                         "0 = auto (only used for small datasets).")
    p.add_argument("--model-dir", default="models")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Column / label auto-detection, so this script isn't locked to one dataset
# ---------------------------------------------------------------------------
URL_COL_CANDIDATES = ["url", "URL", "Url", "link", "Link", "urls"]
LABEL_COL_CANDIDATES = ["status", "label", "Label", "class", "Class", "Result", "type", "Type"]
PHISHING_VALUE_CANDIDATES = ["phishing", "phish", "1", "bad", "malicious", "-1"]
LEGITIMATE_VALUE_CANDIDATES = ["legitimate", "benign", "good", "0", "safe"]


def detect_columns(header_cols, url_col, label_col):
    if url_col is None:
        url_col = next((c for c in URL_COL_CANDIDATES if c in header_cols), None)
        if url_col is None:
            raise ValueError(
                f"Could not auto-detect a URL column in {header_cols}. Pass --url-col explicitly."
            )
    if label_col is None:
        label_col = next((c for c in LABEL_COL_CANDIDATES if c in header_cols), None)
        if label_col is None:
            raise ValueError(
                f"Could not auto-detect a label column in {header_cols}. Pass --label-col explicitly."
            )
    return url_col, label_col


def detect_phishing_value(sample_values, phishing_label):
    if phishing_label is not None:
        return str(phishing_label).strip().lower()
    uniq = {str(v).strip().lower() for v in sample_values.unique()}
    for cand in PHISHING_VALUE_CANDIDATES:
        if cand in uniq:
            return cand
    # Fall back: for a 2-class numeric/text column with no obvious match,
    # assume the less-common label is "phishing" only if exactly 2 classes.
    if len(uniq) == 2:
        counts = sample_values.astype(str).str.strip().str.lower().value_counts()
        guessed = counts.idxmin()
        print(f"  (could not confidently identify the 'phishing' label value; "
              f"guessing '{guessed}' -- pass --phishing-label to override)")
        return guessed
    raise ValueError(
        f"Could not determine which value in the label column means 'phishing'. "
        f"Saw values: {uniq}. Pass --phishing-label explicitly."
    )


# ---------------------------------------------------------------------------
# Data loading: chunked read + optional reservoir sampling so this scales
# from a handful of rows up to files bigger than available RAM.
# ---------------------------------------------------------------------------
def load_dataset(args):
    header = pd.read_csv(args.data_path, nrows=0)
    url_col, label_col = detect_columns(list(header.columns), args.url_col, args.label_col)

    rng = np.random.default_rng(RANDOM_STATE)
    reservoir_urls, reservoir_labels = [], []
    seen = 0
    t0 = time.perf_counter()

    for chunk in pd.read_csv(args.data_path, usecols=[url_col, label_col], chunksize=args.chunksize):
        chunk = chunk.dropna(subset=[url_col, label_col])
        urls = chunk[url_col].astype(str).tolist()
        labels = chunk[label_col].tolist()

        if args.sample_size is None:
            reservoir_urls.extend(urls)
            reservoir_labels.extend(labels)
        else:
            # classic reservoir sampling: keeps a uniform random sample of
            # fixed size without ever holding the full file in memory
            for u, l in zip(urls, labels):
                seen += 1
                if len(reservoir_urls) < args.sample_size:
                    reservoir_urls.append(u)
                    reservoir_labels.append(l)
                else:
                    j = rng.integers(0, seen)
                    if j < args.sample_size:
                        reservoir_urls[j] = u
                        reservoir_labels[j] = l
                continue
        seen = seen if args.sample_size is not None else len(reservoir_urls)
        print(f"  ...read {seen:,} rows so far ({time.perf_counter()-t0:.1f}s)")

    print(f"Loaded {len(reservoir_urls):,} rows from '{args.data_path}' "
          f"(url column: '{url_col}', label column: '{label_col}')")

    labels_series = pd.Series(reservoir_labels)
    phishing_value = detect_phishing_value(labels_series, args.phishing_label)
    y = (labels_series.astype(str).str.strip().str.lower() == phishing_value).astype(int).values
    print(f"  label balance: {int(y.sum()):,} phishing / {int((1-y).sum()):,} legitimate")

    return reservoir_urls, y


def evaluate_model(name, model, X_test, y_test, scale=False, scaler=None):
    X_eval = scaler.transform(X_test) if scale else X_test

    single = X_eval[:1]
    n_reps = 200
    t0 = time.perf_counter()
    for _ in range(n_reps):
        model.predict(single)
    latency_ms = (time.perf_counter() - t0) / n_reps * 1000

    t0 = time.perf_counter()
    y_pred = model.predict(X_eval)
    batch_time_ms = (time.perf_counter() - t0) * 1000
    y_proba = model.predict_proba(X_eval)[:, 1]

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "avg_single_prediction_latency_ms": round(latency_ms, 4),
        "test_set_batch_predict_ms": round(batch_time_ms, 2),
        "test_set_size": len(y_test),
    }


def cross_validated_metrics(name, build_model_fn, X, y, folds, scale=False):
    """Small-dataset path: k-fold CV instead of one noisy holdout split."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        model = build_model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_test, y_proba))

    return {
        "model": name,
        "evaluation": f"{folds}-fold stratified cross-validation (dataset too small for a reliable single split)",
        "accuracy": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "precision": round(float(np.mean(precs)), 4),
        "recall": round(float(np.mean(recs)), 4),
        "f1_score": round(float(np.mean(f1s)), 4),
        "roc_auc": round(float(np.mean(aucs)), 4),
        "test_set_size": f"~{len(y)//folds} per fold x {folds} folds",
    }


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading dataset...")
    urls, y = load_dataset(args)
    n = len(urls)

    print(f"\nExtracting {len(FEATURE_NAMES)} lexical features for {n:,} URLs "
          f"(n_jobs={args.n_jobs})...")
    t0 = time.perf_counter()
    X = extract_features_matrix(urls, n_jobs=args.n_jobs)
    print(f"  done in {time.perf_counter()-t0:.2f}s "
          f"({(time.perf_counter()-t0)/n*1e6:.1f} microseconds/url)")

    use_cv = args.cv_folds > 0 or (args.cv_folds == 0 and n < SMALL_DATASET_THRESHOLD)
    folds = args.cv_folds if args.cv_folds > 0 else 5
    folds = min(folds, max(2, min(np.bincount(y))))  # can't have more folds than the rarer class

    results = {}

    if use_cv:
        print(f"\nDataset has {n:,} rows -- using {folds}-fold cross-validation for honest metrics "
              f"on this small sample, then fitting the final models on ALL the data for deployment.")

        results["logistic_regression"] = cross_validated_metrics(
            "Logistic Regression",
            lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            X, y, folds, scale=True,
        )
        results["xgboost"] = cross_validated_metrics(
            "XGBoost",
            lambda: XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=args.n_jobs or 1,
            ),
            X, y, folds, scale=False,
        )

        # Final deployment models: fit on 100% of the (small) data
        scaler = StandardScaler().fit(X)
        logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(scaler.transform(X), y)
        xgb = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=args.n_jobs or 1,
        ).fit(X, y)

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

        print("\nTraining Logistic Regression...")
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        t0 = time.perf_counter()
        logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        logreg.fit(X_train_scaled, y_train)
        logreg_train_time = time.perf_counter() - t0
        logreg_metrics = evaluate_model("Logistic Regression", logreg, X_test, y_test, scale=True, scaler=scaler)
        logreg_metrics["train_time_sec"] = round(logreg_train_time, 4)
        results["logistic_regression"] = logreg_metrics
        print(json.dumps(logreg_metrics, indent=2))

        print("\nTraining XGBoost...")
        t0 = time.perf_counter()
        # tree_method='hist' scales well to large row counts; depth/estimators
        # scale mildly with dataset size but stay bounded for lightweight inference.
        n_estimators = 150 if n < 200_000 else 250
        xgb = XGBClassifier(
            n_estimators=n_estimators, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=args.n_jobs or 4,
        )
        xgb.fit(X_train, y_train)
        xgb_train_time = time.perf_counter() - t0
        xgb_metrics = evaluate_model("XGBoost", xgb, X_test, y_test, scale=False)
        xgb_metrics["train_time_sec"] = round(xgb_train_time, 4)
        results["xgboost"] = xgb_metrics
        print(json.dumps(xgb_metrics, indent=2))

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\nSelected model for deployment: {best_name}")

    joblib.dump(logreg, f"{args.model_dir}/logistic_regression.joblib")
    joblib.dump(scaler, f"{args.model_dir}/scaler.joblib")
    joblib.dump(xgb, f"{args.model_dir}/xgboost.joblib")

    feature_importance = dict(
        sorted(
            zip(FEATURE_NAMES, [round(float(x), 5) for x in xgb.feature_importances_]),
            key=lambda kv: -kv[1],
        )
    )

    summary = {
        "best_model": best_name,
        "results": results,
        "feature_importance_xgboost": feature_importance,
        "feature_names": FEATURE_NAMES,
        "dataset_size": n,
        "evaluation_mode": "cross_validation" if use_cv else "train_test_split",
        "data_path": args.data_path,
    }
    with open(f"{args.model_dir}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{args.model_dir}/best_model.txt", "w") as f:
        f.write(best_name)

    print(f"\nSaved models + metrics to ./{args.model_dir}/")


if __name__ == "__main__":
    main()
