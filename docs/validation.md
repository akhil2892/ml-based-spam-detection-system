# Validation

The project was tested through its Python API and an actual Chrome browser. Screenshots were opened and visually inspected; interface assessment was not limited to source review.

## Backend and model

`python -m pytest -q`: **68 passed**.

Coverage includes input types and limits, malformed JSON, batch consistency, explicit AI consent, offline operation, reproducible held-out metrics, normalized duplicate separation, additive explanations, artifact checksums, invalid provider keys, malformed provider envelopes, quota and timeout failures, optional reply/rewrite endpoints, and AI disagreement.

`python -m pip check`: no broken requirements. JavaScript syntax and Python import/basic lint checks passed.

The held-out result is **98.55% accuracy**, **96.69% spam precision**, and **91.41% spam recall** on **1,032 messages**. The complete report and baseline tradeoff are in `models/metrics.json` and the README.

## Browser interaction and visual review

`python scripts/browser_check.py --channel chrome`: **18 checks passed**, with no uncaught JavaScript errors.

The browser checked empty input, spam and not-spam verdicts and colors, evidence, keyboard submission, recent history, unfamiliar-language prompts, loading, server errors, edited-message cancellation, batch validation and actual predictions, CSV download and formula protection, HTML injection resistance, and navigation.

Responsive checks cover **1440px desktop**, **768px tablet**, and **390px phone** layouts. All four views were checked for page overflow on tablet and phone. A mobile navigation overflow found during inspection was corrected and rechecked.

Screenshots in `docs/screenshots/` show actual local-model results. Additional test captures and the browser report are generated into ignored `test-results/` when the browser script runs.

An axe-core scan of the four application views and both spam/not-spam result states found **no automated WCAG 2/2.1 A/AA violations** after text-contrast fixes. This automated result does not substitute for a full manual accessibility audit. The interface also provides keyboard focus styles, labels, a skip link, live result messages, and reduced-motion support.

Recorded results: [browser checks](browser-report.json) and [accessibility checks](accessibility-report.json).

## AI verification boundary

AI success, disagreement, invalid output, reply/rewrite, and provider-error paths were checked with controlled responses. Browser AI demonstrations are explicitly simulated. The actual configured Gemini key was rejected as invalid by the provider, so a successful live Gemini review could not be verified. The live check returns an actionable error; local predictions remain functional.

A valid replacement key can be added locally to `.env`; it must not be committed. Re-run `python scripts/check_ai.py` to verify the live provider connection.
