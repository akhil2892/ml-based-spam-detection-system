"""Inference and additive explanations of the linear model's features."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SpamDetector:
    def __init__(self, model_path=None):
        model_path = (
            Path(model_path) if model_path else ROOT / "models" / "spam_pipeline.joblib"
        )
        if not model_path.exists():
            raise RuntimeError(
                "Model missing. Run python train.py before starting the app."
            )
        self.report = json.loads(
            model_path.with_name("metrics.json").read_text(encoding="utf-8")
        )
        if (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            != self.report["artifact_sha256"]
        ):
            raise RuntimeError(
                "Model checksum differs from metrics.json. Retrain to restore matched artifacts."
            )
        bundle = joblib.load(model_path)
        self.pipeline, self.threshold = bundle["pipeline"], bundle["threshold"]
        self.features = self.pipeline.named_steps["features"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.names = self.features.get_feature_names_out()
        self.word_vectorizer = dict(self.features.transformer_list)["word"]

    def predict_many(self, texts):
        start = perf_counter()
        matrix = self.features.transform(texts)
        probabilities = self.classifier.predict_proba(matrix)[:, 1]
        results = []
        for row_index, probability in enumerate(probabilities):
            row = matrix.getrow(row_index)
            contributions = row.data * self.classifier.coef_[0, row.indices]
            ordered = np.argsort(np.abs(contributions))[::-1]
            evidence, word_evidence = [], []
            for index in ordered:
                family, token = self.names[row.indices[index]].split("__", 1)
                item = {
                    "feature": token,
                    "kind": "word" if family == "word" else "character",
                    "direction": "spam" if contributions[index] > 0 else "ham",
                    "contribution": round(float(contributions[index]), 4),
                }
                if len(evidence) < 8:
                    evidence.append(item)
                if family == "word" and len(word_evidence) < 6:
                    word_evidence.append(item)
            tokens = self.word_vectorizer.build_analyzer()(texts[row_index])
            coverage = (
                sum(token in self.word_vectorizer.vocabulary_ for token in tokens)
                / len(tokens)
                if tokens
                else 0.0
            )
            unfamiliar = coverage < 0.2
            predicted = probability >= self.threshold
            reasons = [
                item["feature"]
                for item in word_evidence
                if (item["direction"] == "spam") == predicted
            ][:3]
            results.append(
                {
                    "prediction": "Spam" if predicted else "Not spam",
                    "label": "spam" if predicted else "ham",
                    "spam_probability": round(float(probability), 6),
                    "threshold": round(self.threshold, 2),
                    "needs_review": bool(
                        abs(probability - self.threshold) < 0.12 or unfamiliar
                    ),
                    "unfamiliar_text": unfamiliar,
                    "vocabulary_coverage": round(coverage, 4),
                    "reason": "Learned text features supporting this prediction: "
                    + ", ".join(reasons)
                    + "."
                    if reasons
                    else "The combined word and character patterns drive this prediction.",
                    "evidence": evidence,
                    "word_evidence": word_evidence,
                    "explanation": {
                        "intercept": round(float(self.classifier.intercept_[0]), 6),
                        "feature_contribution_sum": round(
                            float(contributions.sum()), 6
                        ),
                        "log_odds": round(
                            float(self.classifier.decision_function(row)[0]), 6
                        ),
                    },
                    "source": "local_ml",
                    "model_version": self.report["model_version"],
                }
            )
        for result in results:
            result["processing_time_ms"] = round((perf_counter() - start) * 1000, 2)
        return results

    def predict(self, text):
        return self.predict_many([text])[0]
