import pytest

HAM = "Hey, are we still meeting for coffee tomorrow? I can be there around 6. Let me know."
SPAM = "Congratulations! You have won a cash prize. Call 09061701461 now to claim your FREE reward!"


def test_local_predictions_and_batch_are_consistent(client):
    single = [
        client.post("/predict", json={"text": text}).get_json() for text in (HAM, SPAM)
    ]
    assert [row["label"] for row in single] == ["ham", "spam"]
    batch = client.post("/predict_batch", json={"texts": [HAM, SPAM]}).get_json()
    assert batch["summary"] == {"total": 2, "spam": 1, "ham": 1}
    for expected, actual in zip(single, batch["results"]):
        for key in ("prediction", "spam_probability", "threshold", "evidence"):
            assert actual[key] == expected[key]
    assert all(row["source"] == "local_ml" for row in single)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"text": ""},
        {"text": "   "},
        {"text": 3},
        {"text": True},
        {"text": []},
        {"text": {}},
        {"text": "a" * 5001},
    ],
)
def test_invalid_single_requests(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert isinstance(response.get_json()["error"], str)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"texts": ""},
        {"texts": []},
        {"texts": [HAM] * 51},
        {"texts": [HAM, None]},
        {"texts": [HAM, ""]},
        {"texts": ["x" * 5001]},
    ],
)
def test_invalid_batches_are_atomic(client, payload):
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 400
    assert "results" not in response.get_json()


def test_batch_limit_and_character_limit(client):
    assert client.post("/predict_batch", json={"texts": [HAM] * 50}).status_code == 200
    assert (
        client.post("/predict", json={"text": "hello " * 833 + "hi"}).status_code == 200
    )


def test_malformed_and_non_json_input(client):
    for body, content_type in [
        ("not-json", "application/json"),
        ("text=hello", "text/plain"),
    ]:
        assert (
            client.post("/predict", data=body, content_type=content_type).status_code
            == 400
        )


def test_body_limit_and_json_http_errors(client):
    assert (
        client.post(
            "/predict", data="x" * 300001, content_type="application/json"
        ).status_code
        == 413
    )
    assert client.get("/predict").status_code == 405
    assert client.get("/missing").get_json()["error"]


def test_health_page_and_model_report(client, detector):
    health = client.get("/health").get_json()
    assert health["status"] == "healthy" and not health["ai_configured"]
    report = client.get("/api/model").get_json()
    assert report == detector.report
    assert sum(report["test"]["confusion_matrix"].values()) == report["split"]["test"]
    page = client.get("/")
    assert page.status_code == 200
    assert b"Message analyzer" in page.data
    assert "frame-ancestors" in page.headers["Content-Security-Policy"]
    assert (
        client.post("/predict", json={"text": HAM}).headers["Cache-Control"]
        == "no-store"
    )


@pytest.mark.parametrize("endpoint", ["/ai_review", "/generate_reply", "/polish_text"])
@pytest.mark.parametrize("consent", [None, False, "true", 1])
def test_ai_requires_explicit_boolean_consent(client, endpoint, consent):
    assert (
        client.post(endpoint, json={"text": HAM, "consent": consent}).status_code == 400
    )


@pytest.mark.parametrize("endpoint", ["/ai_review", "/generate_reply", "/polish_text"])
def test_ai_disabled_has_clear_errors_and_keeps_local_working(client, endpoint):
    assert client.post(endpoint, json={"text": HAM, "consent": True}).status_code == 503
    assert client.post("/predict", json={"text": HAM}).status_code == 200
