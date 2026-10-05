import pytest

from app import create_app
from spam_detector.ai import GeminiReviewer
from spam_detector.predictor import SpamDetector


@pytest.fixture(scope="session")
def detector():
    return SpamDetector()


@pytest.fixture
def client(detector):
    return create_app(
        {"TESTING": True}, detector=detector, reviewer=GeminiReviewer()
    ).test_client()
