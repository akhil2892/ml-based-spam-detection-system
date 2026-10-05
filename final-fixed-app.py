"""Compatibility entry point; the maintained application lives in app.py."""

import os

from app import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
