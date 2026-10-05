# Model card

**Task:** binary English SMS spam classification. Labels: `ham` and `spam`.

**Intended use:** educational demonstration, local message triage, and inspection of learned text features. Not validated as a standalone email security filter or a fact-checking system.

**Data:** UCI SMS Spam Collection, Almeida and Hidalgo (2011), [DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84), CC BY 4.0. Raw messages are not redistributed here. Download URL and SHA-256 are stored in `models/metrics.json`.

**Preprocessing:** Unicode NFKC, case folding and whitespace collapse are used to identify duplicates. Conflicting-label groups are removed. Original text is used for vectorization. TF-IDF lowercases text, strips Unicode accents, uses sublinear TF, and excludes features seen in fewer than two training documents. Character fragments use `char_wb` boundaries.

**Representation:** 8,181 word unigrams/bigrams and 27,164 character 3–5 grams. Feature-union weights: word 1.0, character 0.65. Feature counts may change with retraining.

**Estimator:** logistic regression, balanced class weights, `liblinear` solver, `C=8`, `max_iter=2000`, seed 42. A validation-selected spam threshold of 0.57 replaces the usual 0.5 default.

**Evaluation design:** 64% train, 16% validation, 20% test after normalized deduplication. Three C values and 51 thresholds are evaluated on validation data. Highest validation spam F1 wins; precision and closeness to 0.5 break ties. Neither TF-IDF vocabulary nor classifier is fitted on validation/test messages. The final artifact retains the same training-only fit used for its held-out report.

**Results:** on 1,032 test messages, accuracy 98.55%, spam precision 96.69%, spam recall 91.41%, spam F1 93.98%, average precision 98.66%, Brier score 0.010900. Confusion counts: 900 correct ham, 117 correct spam, 4 false alarms, 11 missed spam. One split is not an estimate of universal performance; no confidence interval or multi-seed study is claimed.

**Baseline:** CountVectorizer and MultinomialNB, retrained on the same training split. Accuracy is equal, precision is slightly higher for the baseline, and recall is slightly higher for the hybrid model. The F1 difference is small. The original supplied model had a 302-feature vocabulary and no dataset or split provenance, so it is not used to establish historical benchmark performance.

**Score interpretation:** the output is an uncalibrated logistic model probability. Balanced training and a tuned threshold affect its meaning. The UI calls it a model score; it does not add heuristic confidence points or estimate the probability that a link is safe.

**Explanations:** feature values multiplied by coefficients give additive log-odds contributions. The API returns the complete sum and intercept, plus selected features. The UI prioritizes word features for readability. Contributions explain this model's decision, not causal properties of spam.

**Review prompts:** flag scores within 0.12 of the threshold and word-vocabulary coverage below 20%. These are transparent interface heuristics. Character features may still match unfamiliar words; low coverage is not proof of out-of-distribution input, and high coverage does not prove reliability.

**Known limits:** historical English SMS; domain and language shift; similar spam templates across splits; no modern phishing benchmark; no grounding for factual claims; no probability calibration study. Not-spam predictions can still describe scams, unsafe links, or inaccurate claims.

**AI separation:** Gemini is opt-in and independent. Its verdict, explanations, and availability do not affect local predictions, confidence, training, or evaluation. Prompt instructions and schemas reduce malformed responses, but an LLM second opinion can still be wrong.
