Training downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) here on first use. Raw data is excluded from Git; the archive URL and SHA-256 are recorded in `models/metrics.json`.

Credit: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. DOI: [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The UCI page lists 5,574 instances; the downloaded file's actual parsed count is recorded in the report. Training normalizes Unicode, case and whitespace for deduplication, then removes groups with contradictory labels. Original message text is retained for vectorization. This avoids exact duplicate leakage but does not guarantee separation of similar templates.

Custom data: `python train.py --data path/to/messages.tsv`. Each line must contain `ham` or `spam`, a tab, and the message. This replaces the bundled model. Benchmark metrics then refer to your dataset, not UCI.
