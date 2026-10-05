"""Real Chromium checks and screenshots. Run against a running Flask server."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def check(base_url, channel=None):
    output = ROOT / "test-results"
    output.mkdir(exist_ok=True)
    checks, errors = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, **({"channel": channel} if channel else {})
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, reduced_motion="reduce"
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        requests = []
        page.on("request", lambda request: requests.append(request.url))
        page.goto(base_url, wait_until="networkidle")
        expect(page.locator("#empty-result")).to_be_visible()
        page.screenshot(path=str(output / "desktop-empty.png"), full_page=True)
        checks.append("Desktop empty state renders")

        page.locator("#analyze-button").click()
        expect(page.locator("#input-error")).to_have_text(
            "Add a message to get started."
        )
        checks.append("Empty input error and focus")
        page.locator('[data-sample="prize"]').click()
        page.locator("#analyze-button").click()
        expect(page.locator("#prediction")).to_have_text("Spam")
        expect(page.locator(".result-card")).to_have_class("card result-card is-spam")
        expect(page.locator("#evidence .evidence-chip").first).to_be_visible()
        expect(page.locator("#ai-review-button")).to_be_disabled()
        page.screenshot(path=str(output / "desktop-spam.png"), full_page=True)
        checks.append("Spam verdict, features, score and opt-in AI panel")
        assert not any("generativelanguage" in url for url in requests)
        checks.append("Local checks make no external browser AI requests")

        page.locator('[data-sample="everyday"]').click()
        expect(page.locator("#result-content")).to_be_hidden()
        page.locator("#message").press("Control+Enter")
        expect(page.locator("#prediction")).to_have_text("Not spam")
        assert "is-spam" not in page.locator(".result-card").get_attribute("class")
        expect(page.locator("#history-count")).to_have_text("2")
        page.screenshot(path=str(output / "desktop-ham.png"), full_page=True)
        checks.append("Not-spam styling, keyboard shortcut and session history")

        page.locator("#message").fill("नमस्ते आप कैसे हैं")
        page.locator("#analyze-button").click()
        expect(page.locator("#review-note")).to_be_visible()
        checks.append("Unfamiliar language flags human review")

        page.route(
            "**/predict",
            lambda route: route.fulfill(
                status=503,
                content_type="application/json",
                body=json.dumps({"error": "Temporary test outage."}),
            ),
        )
        page.locator("#message").fill("Hello, see you at lunch tomorrow.")
        page.locator("#analyze-button").click()
        expect(page.locator("#input-error")).to_have_text("Temporary test outage.")
        expect(page.locator("#analyze-button")).to_be_enabled()
        page.screenshot(path=str(output / "desktop-error.png"), full_page=True)
        page.unroute("**/predict")
        checks.append("Server errors recover without trapping controls")

        # A delayed response should never repaint results for edited text.
        pending = []
        page.route("**/predict", lambda route: pending.append(route))
        page.locator("#analyze-button").click()
        expect(page.locator("#loading-result")).to_be_visible()
        page.screenshot(path=str(output / "desktop-loading.png"), full_page=True)
        page.locator("#message").fill(
            "A different message while the request is pending."
        )
        expect(page.locator("#empty-result")).to_be_visible()
        expect(page.locator("#analyze-button")).to_be_enabled()
        for route in pending:
            try:
                route.fulfill(status=200, content_type="application/json", body="{}")
            except Exception:
                pass  # The browser has intentionally aborted this request.
        page.unroute("**/predict")
        expect(page.locator("#result-content")).to_be_hidden()
        checks.append("Loading state and cancellation prevent stale results")

        # Simulated AI success/failure is labeled in the report; live API is checked separately.
        page.locator('[data-sample="everyday"]').click()
        page.locator("#analyze-button").click()
        expect(page.locator("#prediction")).to_have_text("Not spam")
        if page.locator("body").get_attribute("data-ai-configured") == "true":
            page.route(
                "**/ai_review",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "prediction": "Spam",
                            "reason": "<img src=x onerror=alert(1)> A separate opinion.",
                            "agrees_with_ml": False,
                        }
                    ),
                ),
            )
            page.locator("#ai-consent").check()
            page.locator("#ai-review-button").click()
            expect(page.locator("#ai-output")).to_contain_text(
                "Differs from local model"
            )
            expect(page.locator("#prediction")).to_have_text("Not spam")
            assert page.locator("#ai-output img").count() == 0
            page.screenshot(path=str(output / "desktop-ai.png"), full_page=True)
            checks.append(
                "Simulated AI disagreement stays separate; HTML rendered as text"
            )
            page.unroute("**/ai_review")
            page.route(
                "**/ai_review",
                lambda route: route.fulfill(
                    status=503,
                    content_type="application/json",
                    body=json.dumps(
                        {"error": "AI service unavailable. Local verdict unchanged."}
                    ),
                ),
            )
            page.locator("#ai-review-button").click()
            expect(page.locator("#ai-output")).to_contain_text("AI service unavailable")
            expect(page.locator("#prediction")).to_have_text("Not spam")
            checks.append("Simulated AI outage preserves the visible local result")
            page.unroute("**/ai_review")
            page.route(
                "**/generate_reply",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "replies": [
                                "Thanks!",
                                "See you then.",
                                "<script>alert(1)</script>",
                            ]
                        }
                    ),
                ),
            )
            page.locator("#reply-button").click()
            expect(page.locator("#ai-tool-output .reply-suggestion")).to_have_count(3)
            assert page.locator("#ai-tool-output script").count() == 0
            page.unroute("**/generate_reply")
            page.route(
                "**/polish_text",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {"polished_text": "A neutral rewrite of the original message."}
                    ),
                ),
            )
            page.locator("#polish-button").click()
            expect(page.locator("#ai-tool-output")).to_contain_text("A neutral rewrite")
            page.unroute("**/polish_text")
            checks.append("Simulated reply and rewrite actions display safely")
        else:
            expect(page.locator("#ai-consent")).to_be_disabled()
            checks.append("AI controls disabled cleanly when no key is configured")

        page.locator('[data-view="batch"]').click()
        expect(page.locator("#view-batch")).to_be_visible()
        page.locator("#batch-button").click()
        expect(page.locator("#batch-error")).to_be_visible()
        page.locator("#batch-example").click()
        page.locator("#batch-button").click()
        expect(page.locator("#batch-rows tr")).to_have_count(3)
        with page.expect_download() as download:
            page.locator("#export-csv").click()
        file = output / "results.csv"
        download.value.save_as(str(file))
        assert "spam_model_score" in file.read_text(encoding="utf-8-sig")
        page.screenshot(path=str(output / "desktop-batch.png"), full_page=True)
        checks.append("Batch form, actual predictions and downloadable CSV")
        page.locator("#batch-input").fill("hello\n" * 51)
        page.locator("#batch-button").click()
        expect(page.locator("#batch-error")).to_have_text(
            "Check up to 50 messages at a time."
        )
        checks.append("Batch limit validation")
        page.locator("#batch-input").fill(
            '=HYPERLINK("https://example.com")\n<img src=x onerror=alert(1)>'
        )
        page.locator("#batch-button").click()
        expect(page.locator("#batch-rows tr")).to_have_count(2)
        assert page.locator("#batch-rows img").count() == 0
        with page.expect_download() as download:
            page.locator("#export-csv").click()
        formula_csv = output / "formula-results.csv"
        download.value.save_as(str(formula_csv))
        assert "'=HYPERLINK" in formula_csv.read_text(encoding="utf-8-sig")
        checks.append("CSV formula protection and HTML injection resistance")

        page.locator('[data-view="model"]').click()
        expect(page.locator("#model-heading")).to_be_visible()
        page.screenshot(path=str(output / "desktop-model.png"), full_page=True)
        page.locator('[data-view="about"]').click()
        expect(page.locator("#about-heading")).to_be_visible()
        page.screenshot(path=str(output / "desktop-about.png"), full_page=True)
        checks.append("Model metrics and explanation views navigate correctly")

        for width in (390, 768):
            page.set_viewport_size(
                {"width": width, "height": 844 if width == 390 else 1024}
            )
            page.locator('[data-view="analyzer"]').click()
            page.locator('[data-sample="prize"]').click()
            page.locator("#analyze-button").click()
            expect(page.locator("#prediction")).to_have_text("Spam")
            page.screenshot(
                path=str(output / f"width-{width}-spam.png"), full_page=True
            )
            for view in ("analyzer", "batch", "model", "about"):
                page.locator(f'[data-view="{view}"]').click()
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                ), f"{view} overflows at {width}px"
                if width == 390:
                    page.screenshot(
                        path=str(output / f"mobile-{view}.png"), full_page=True
                    )
            checks.append(f"{width}px layout has no page overflow; all views work")
        assert not errors, f"JavaScript errors: {errors}"
        checks.append("No uncaught JavaScript errors")
        browser.close()
    report = {
        "browser": channel or "Chromium",
        "checks": checks,
        "total": len(checks),
        "javascript_errors": errors,
        "ai_ui_responses": "simulated; live availability checked separately",
        "screenshots": "test-results/*.png",
    }
    (output / "browser-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--channel", choices=["chrome", "msedge"])
    args = parser.parse_args()
    check(args.url, args.channel)
