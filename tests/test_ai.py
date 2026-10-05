import json
from unittest.mock import Mock

import pytest
import requests

from app import create_app
from spam_detector.ai import AIUnavailable, GeminiReviewer

TEXT = "Please send your notes for the meeting tomorrow. Thanks!"


def fake_response(data):
    response = Mock(status_code=200)
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(data)}]}}]
    }
    return response


def test_ai_disagreement_does_not_override_ml(detector, monkeypatch):
    local = detector.predict(TEXT)
    opposite = "Spam" if local["label"] == "ham" else "Not spam"
    call = Mock(
        return_value=fake_response(
            {"prediction": opposite, "reason": "A separate contextual opinion."}
        )
    )
    monkeypatch.setattr(requests, "post", call)
    client = create_app(
        {"TESTING": True}, detector, GeminiReviewer("test-only-secret")
    ).test_client()
    assert (
        client.post("/predict", json={"text": TEXT}).get_json()["prediction"]
        == local["prediction"]
    )
    call.assert_not_called()
    review = client.post("/ai_review", json={"text": TEXT, "consent": True}).get_json()
    assert review["prediction"] == opposite and not review["agrees_with_ml"]
    assert (
        client.post("/predict", json={"text": TEXT}).get_json()["prediction"]
        == local["prediction"]
    )
    assert call.call_count == 1
    args = call.call_args.kwargs
    assert args["timeout"] == (5, 20)
    assert args["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert args["headers"]["x-goog-api-key"] == "test-only-secret"
    assert "test-only-secret" not in call.call_args.args[0]


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"prediction": "SAFE", "reason": "ok"},
        {"prediction": "Spam", "reason": 123},
        {"prediction": "Spam"},
        {"prediction": "Spam", "reason": ""},
    ],
)
def test_invalid_provider_reviews_are_rejected(monkeypatch, data):
    monkeypatch.setattr(requests, "post", Mock(return_value=fake_response(data)))
    with pytest.raises(AIUnavailable):
        GeminiReviewer("test-key").review(TEXT, {"prediction": "Not spam"})


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("secret-key"),
        requests.ConnectionError("private message"),
        requests.HTTPError("credential"),
    ],
)
def test_provider_errors_are_redacted(detector, monkeypatch, error):
    monkeypatch.setattr(requests, "post", Mock(side_effect=error))
    client = create_app(
        {"TESTING": True}, detector, GeminiReviewer("test-key")
    ).test_client()
    response = client.post("/ai_review", json={"text": TEXT, "consent": True})
    assert response.status_code == 503
    assert "secret" not in response.get_json()["error"]
    assert "private" not in response.get_json()["error"]
    assert "credential" not in response.get_json()["error"]


def test_quota_error_is_handled(monkeypatch):
    monkeypatch.setattr(requests, "post", Mock(return_value=Mock(status_code=429)))
    with pytest.raises(AIUnavailable) as result:
        GeminiReviewer("test-key").review(TEXT, {"prediction": "Not spam"})
    assert result.value.status == 429


def test_invalid_key_has_an_actionable_redacted_error(monkeypatch):
    response = Mock(status_code=400)
    response.json.return_value = {
        "error": {"message": "API key not valid. private-secret"}
    }
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    with pytest.raises(AIUnavailable, match="Check GEMINI_API_KEY") as result:
        GeminiReviewer("test-key").review(TEXT, {"prediction": "Not spam"})
    assert "private-secret" not in str(result.value)


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"candidates": []},
        {"candidates": [{"content": {"parts": ["bad structure"]}}]},
    ],
)
def test_invalid_provider_envelope_is_a_service_error(monkeypatch, envelope):
    response = Mock(status_code=200)
    response.json.return_value = envelope
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    with pytest.raises(AIUnavailable):
        GeminiReviewer("test-key").review(TEXT, {"prediction": "Not spam"})


def test_invalid_prediction_type_is_a_service_error(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        Mock(return_value=fake_response({"prediction": [], "reason": "text"})),
    )
    with pytest.raises(AIUnavailable):
        GeminiReviewer("test-key").review(TEXT, {"prediction": "Not spam"})


def test_process_ai_request_limit(monkeypatch):
    call = Mock(
        return_value=fake_response({"prediction": "Not spam", "reason": "Conversation"})
    )
    monkeypatch.setattr(requests, "post", call)
    reviewer = GeminiReviewer("test-key")
    for _ in range(10):
        reviewer.review(TEXT, {"prediction": "Not spam"})
    with pytest.raises(AIUnavailable) as result:
        reviewer.review(TEXT, {"prediction": "Not spam"})
    assert result.value.status == 429 and call.call_count == 10


def test_reply_and_rewrite_endpoints(detector, monkeypatch):
    call = Mock(
        side_effect=[
            fake_response(
                {"replies": ["Thanks!", "See you soon.", "I will send them."]}
            ),
            fake_response({"polished_text": "Please review the offer."}),
        ]
    )
    monkeypatch.setattr(requests, "post", call)
    client = create_app(
        {"TESTING": True}, detector, GeminiReviewer("test-key")
    ).test_client()
    assert (
        client.post("/generate_reply", json={"text": TEXT, "consent": True}).get_json()[
            "replies"
        ][0]
        == "Thanks!"
    )
    assert (
        client.post("/polish_text", json={"text": TEXT, "consent": True}).get_json()[
            "polished_text"
        ]
        == "Please review the offer."
    )
    spam = "Congratulations! You won a FREE cash prize. Call now to claim your reward!"
    assert (
        client.post("/generate_reply", json={"text": spam, "consent": True}).status_code
        == 422
    )
    assert call.call_count == 2


@pytest.mark.parametrize(
    "data",
    [{"replies": "hello"}, {"replies": ["one"]}, {"replies": ["one", {}, "three"]}],
)
def test_invalid_reply_schema(monkeypatch, data):
    monkeypatch.setattr(requests, "post", Mock(return_value=fake_response(data)))
    with pytest.raises(AIUnavailable):
        GeminiReviewer("test-key").replies(TEXT)
