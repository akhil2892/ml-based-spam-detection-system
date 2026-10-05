import json
import math

import pytest

from spam_detector.predictor import SpamDetector
from spam_detector.training import canonical, load_data, split_data


@pytest.mark.parametrize(
    "text",
    [
        "hello there, see you at lunch",
        "FREE cash prize! Call now to claim your reward!",
        '<script>alert("x")</script>',
        "नमस्ते आप कैसे हैं",
        "👋 🌻 🐈",
    ],
)
def test_explanations_reconstruct_model_score(detector, text):
    result = detector.predict(text)
    explanation = result["explanation"]
    assert (
        abs(
            explanation["intercept"]
            + explanation["feature_contribution_sum"]
            - explanation["log_odds"]
        )
        < 2e-6
    )
    assert math.isclose(
        1 / (1 + math.exp(-explanation["log_odds"])),
        result["spam_probability"],
        abs_tol=1e-6,
    )
    assert (result["label"] == "spam") == (
        result["spam_probability"] >= result["threshold"]
    )


def test_unfamiliar_text_is_flagged(detector):
    result = detector.predict("नमस्ते आप कैसे हैं")
    assert result["unfamiliar_text"] and result["needs_review"]


def test_artifact_checksum_is_checked(tmp_path, detector):
    root = tmp_path / "models"
    root.mkdir()
    (root / "spam_pipeline.joblib").write_bytes(b"tampered")
    (root / "metrics.json").write_text(json.dumps(detector.report))
    with pytest.raises(RuntimeError, match="checksum"):
        SpamDetector(root / "spam_pipeline.joblib")


def test_normalized_duplicates_and_conflicts_are_removed(tmp_path):
    path = tmp_path / "messages.tsv"
    path.write_text(
        "ham\tHello  friend\nham\thello friend\nspam\tFREE cash\nham\tfree CASH\nham\tLunch tomorrow\n",
        encoding="utf-8",
    )
    texts, labels, info = load_data(path)
    assert list(texts) == ["Hello  friend", "Lunch tomorrow"]
    assert list(labels) == [0, 0]
    assert info["duplicate_rows_removed"] == 2
    assert info["conflicting_groups_removed"] == 1
    assert info["source"] == "Custom labeled dataset"
    assert info["citation"] is None


def test_missing_custom_data_is_not_replaced_with_uci(tmp_path):
    with pytest.raises(FileNotFoundError, match="Custom dataset"):
        load_data(tmp_path / "missing.tsv")


def test_holdout_split_has_no_normalized_overlap():
    texts, labels, info = load_data()
    splits = split_data(texts, labels)
    groups = [{canonical(text) for text in texts[index]} for index in splits]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum(len(index) for index in splits) == info["unique_rows"]


def test_test_metrics_are_reproducible(detector):
    from spam_detector.training import metrics

    texts, labels, dataset = load_data()
    assert dataset["sha256"] == detector.report["dataset"]["sha256"]
    _, _, test_idx = split_data(texts, labels)
    score = metrics(
        labels[test_idx],
        detector.pipeline.predict_proba(texts[test_idx])[:, 1],
        detector.threshold,
    )
    assert score == detector.report["test"]
