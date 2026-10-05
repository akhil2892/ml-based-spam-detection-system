"""Local ML predictions with separate, explicit AI assistance."""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from spam_detector.ai import AIUnavailable, GeminiReviewer
from spam_detector.predictor import SpamDetector

ROOT = Path(__file__).resolve().parent
MAX_TEXT_LENGTH, MAX_BATCH_SIZE = 5000, 50


def create_app(config=None, detector=None, reviewer=None):
    load_dotenv(ROOT / ".env")
    application = Flask(__name__)
    application.config.update(MAX_CONTENT_LENGTH=300_000)
    if config:
        application.config.update(config)
    detector = detector or SpamDetector()
    reviewer = reviewer or GeminiReviewer(
        os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    application.extensions.update(spam_detector=detector, ai_reviewer=reviewer)

    def payload():
        if not request.is_json:
            raise ValueError("Send a JSON object with Content-Type: application/json.")
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("The request must contain a valid JSON object.")
        return data

    def valid_text(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a nonempty message.")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Messages must be {MAX_TEXT_LENGTH:,} characters or fewer."
            )
        return text.strip()

    def ai_payload():
        data = payload()
        text = valid_text(data.get("text"))
        if data.get("consent") is not True:
            raise ValueError(
                "AI assistance requires consent: true to send this message to Gemini."
            )
        return text

    @application.get("/")
    def home():
        return render_template(
            "index.html", report=detector.report, ai_configured=reviewer.configured
        )

    @application.get("/health")
    def health():
        return jsonify(
            status="healthy",
            model="loaded",
            model_version=detector.report["model_version"],
            ai_configured=reviewer.configured,
        )

    @application.get("/api/model")
    def model_info():
        return jsonify(detector.report)

    @application.post("/predict")
    def predict():
        return jsonify(detector.predict(valid_text(payload().get("text"))))

    @application.post("/predict_batch")
    def predict_batch():
        texts = payload().get("texts")
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
            raise ValueError(
                f"Provide between 1 and {MAX_BATCH_SIZE} messages in texts."
            )
        results = detector.predict_many([valid_text(text) for text in texts])
        return jsonify(
            results=results,
            summary={
                "total": len(results),
                "spam": sum(row["label"] == "spam" for row in results),
                "ham": sum(row["label"] == "ham" for row in results),
            },
        )

    @application.post("/ai_review")
    def ai_review():
        text = ai_payload()
        return jsonify(reviewer.review(text, detector.predict(text)))

    @application.post("/generate_reply")
    def generate_reply():
        text = ai_payload()
        if detector.predict(text)["label"] == "spam":
            return jsonify(
                error="Reply suggestions are available for messages classified as not spam."
            ), 422
        return jsonify(reviewer.replies(text))

    @application.post("/polish_text")
    def polish_text():
        return jsonify(reviewer.polish(ai_payload()))

    @application.errorhandler(ValueError)
    def input_error(error):
        return jsonify(error=str(error)), 400

    @application.errorhandler(AIUnavailable)
    def ai_error(error):
        return jsonify(error=str(error), source="ai_second_opinion"), error.status

    @application.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @application.errorhandler(Exception)
    def unexpected_error(error):
        application.logger.error(
            "Unhandled application error (%s)", type(error).__name__
        )
        return jsonify(
            error="This request could not be completed. Please try again."
        ), 500

    @application.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["Cache-Control"] = (
            "no-store" if request.path != "/" else "no-cache"
        )
        return response

    return application


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
