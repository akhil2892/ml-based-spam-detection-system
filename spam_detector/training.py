"""Reproducible training: vocabulary and threshold never see the test split."""

import hashlib
import io
import json
import unicodedata
import zipfile
from pathlib import Path

import joblib
import numpy as np
import requests
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SEED = 42


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_data(path=None):
    is_uci = path is None
    path = Path(path) if path else ROOT / "data" / "SMSSpamCollection"
    if not path.exists():
        if not is_uci:
            raise FileNotFoundError(f"Custom dataset not found: {path}")
        response = requests.get(DATA_URL, timeout=(15, 60))
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read("SMSSpamCollection"))
            (path.parent / "source-readme.txt").write_bytes(archive.read("readme"))
    groups = {}
    raw_rows = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        label, text = line.split("\t", 1)
        if label not in {"ham", "spam"} or not text.strip():
            raise ValueError("Expected a nonempty ham/spam labeled TSV dataset.")
        raw_rows += 1
        group = groups.setdefault(
            canonical(text), {"text": text.strip(), "labels": set()}
        )
        group["labels"].add(int(label == "spam"))
    clean = [g for g in groups.values() if len(g["labels"]) == 1]
    texts = np.array([g["text"] for g in clean])
    labels = np.array([next(iter(g["labels"])) for g in clean])
    info = {
        "source": "UCI SMS Spam Collection" if is_uci else "Custom labeled dataset",
        "url": DATA_URL if is_uci else None,
        "citation": "Almeida, T. & Hidalgo, J. (2011). https://doi.org/10.24432/C5CC84"
        if is_uci
        else None,
        "license": "CC BY 4.0" if is_uci else "Provided by dataset owner",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": raw_rows,
        "unique_rows": len(clean),
        "duplicate_rows_removed": raw_rows - len(groups),
        "conflicting_groups_removed": sum(
            len(g["labels"]) > 1 for g in groups.values()
        ),
        "class_counts": {
            "ham": int((labels == 0).sum()),
            "spam": int((labels == 1).sum()),
        },
    }
    return texts, labels, info


def split_data(texts, labels):
    train_val, test = train_test_split(
        np.arange(len(texts)), test_size=0.2, stratify=labels, random_state=SEED
    )
    train, validation = train_test_split(
        train_val, test_size=0.2, stratify=labels[train_val], random_state=SEED
    )
    return train, validation, test


def hybrid_pipeline(c=4):
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    min_df=2,
                    max_df=0.98,
                    max_features=20000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    min_df=2,
                    max_features=30000,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 0.65},
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def baseline_pipeline():
    return Pipeline(
        [("features", CountVectorizer()), ("classifier", MultinomialNB(alpha=1.0))]
    )


def metrics(labels, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(labels, predicted)), 6),
        "spam_precision": round(
            float(precision_score(labels, predicted, zero_division=0)), 6
        ),
        "spam_recall": round(
            float(recall_score(labels, predicted, zero_division=0)), 6
        ),
        "spam_f1": round(float(f1_score(labels, predicted, zero_division=0)), 6),
        "average_precision": round(
            float(average_precision_score(labels, probabilities)), 6
        ),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "confusion_matrix": {
            "true_ham": int(tn),
            "false_spam": int(fp),
            "missed_spam": int(fn),
            "true_spam": int(tp),
        },
        "samples": len(labels),
    }


def train(data_path=None):
    texts, labels, dataset = load_data(data_path)
    train_idx, val_idx, test_idx = split_data(texts, labels)
    baseline = baseline_pipeline().fit(texts[train_idx], labels[train_idx])
    candidates = []
    for c in (2, 4, 8):
        pipeline = hybrid_pipeline(c).fit(texts[train_idx], labels[train_idx])
        probabilities = pipeline.predict_proba(texts[val_idx])[:, 1]
        for threshold in np.linspace(0.25, 0.75, 51):
            score = metrics(labels[val_idx], probabilities, float(threshold))
            candidates.append(
                (
                    score["spam_f1"],
                    score["spam_precision"],
                    -abs(threshold - 0.5),
                    c,
                    float(threshold),
                    pipeline,
                    score,
                )
            )
    selected = max(candidates, key=lambda row: row[:3])
    _, _, _, c, threshold, pipeline, validation_metrics = selected
    report = {
        "model_name": "Word + character TF-IDF / Logistic regression",
        "model_version": "1.0",
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "dataset": dataset,
        "split": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "split_strategy": "64/16/20 stratified split after NFKC/case/whitespace deduplication",
        "selection": {
            "objective": "validation spam F1; precision then proximity to 0.5 break ties",
            "C": c,
            "threshold": round(threshold, 2),
            "candidates": len(candidates),
        },
        "validation": validation_metrics,
        "test": metrics(
            labels[test_idx], pipeline.predict_proba(texts[test_idx])[:, 1], threshold
        ),
        "baseline": {
            "name": "CountVectorizer + MultinomialNB",
            "threshold": 0.5,
            "test": metrics(
                labels[test_idx], baseline.predict_proba(texts[test_idx])[:, 1]
            ),
        },
        "candidate_validation": [
            {
                "C": value,
                "best_spam_f1": max(row[0] for row in candidates if row[3] == value),
            }
            for value in (2, 4, 8)
        ],
        "features": {
            name: len(vectorizer.get_feature_names_out())
            for name, vectorizer in pipeline.named_steps["features"].transformer_list
        },
        "probability_note": "Uncalibrated model score, not a guarantee or safety rating. Class weighting and threshold tuning affect its interpretation.",
        "limitations": [
            "English SMS corpus; modern phishing, long emails, other languages and domain shifts are not validated.",
            "Exact normalized duplicates are removed; similar spam templates may still occur across splits.",
            "A not-spam prediction does not establish that links, claims or senders are safe.",
            "The original model has no training provenance, so this baseline is retrained on the same split rather than presented as its historical accuracy.",
        ],
    }
    output = ROOT / "models"
    output.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "threshold": threshold},
        output / "spam_pipeline.joblib",
        compress=3,
    )
    report["artifact_sha256"] = hashlib.sha256(
        (output / "spam_pipeline.joblib").read_bytes()
    ).hexdigest()
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": report["selection"],
                "test": report["test"],
                "baseline": report["baseline"]["test"],
            },
            indent=2,
        )
    )
    return report
