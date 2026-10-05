"""One live smoke check with a synthetic message. Never prints credentials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

from dotenv import load_dotenv

from spam_detector.ai import AIUnavailable, GeminiReviewer

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    reviewer = GeminiReviewer(
        os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    print(f"AI configured: {reviewer.configured}; model: {reviewer.model}")
    try:
        review = reviewer.review(
            "Hey, are we still meeting for coffee tomorrow?", {"prediction": "Not spam"}
        )
        print(f"Live review succeeded: {review['prediction']}")
    except AIUnavailable as error:
        print(f"Live review unavailable ({error.status}): {error}")
        sys.exit(1)
