Foundations of AI — Lab 2: Weather Prediction

Overview

This repository contains my implementation for Weather Prediction. The goal is to predict three booleans for Rochester (ROC):
- Whether today's temperature will be higher than yesterday's
- Whether today's average temperature will be higher than the long-term average
- Whether there will be more than a trace of precipitation today

The project uses CF6 reports from the National Weather Service as raw input, extracts daily features, and trains decision tree and random forest models.

Repository structure

- data/                # (NOT included) historical monthly CF6 text files (large)
- daily_data/          # (NOT included) daily CF6 text files used for prediction examples
- models/              # (NOT included) saved model files (.pkl) and feature importance plots
- data_collection.py   # Fetches CF6 reports from NWS
- data_processing.py   # Parses CF6 text, builds DataFrame and train/test features
- weather_predictor.py # Prediction API: predict(modeltype, day5,day4,day3,day2,day1)
- train.py             # Script to run data collection, training and evaluation
- models.py            # Decision tree and RandomForest implementations + train/evaluate
- run_tests.py         # (if present) tests / utilities
- writeup.pdf          # Writeup describing data, features, and training processes

How to run (development)

1) Create and activate a Python 3.10+ virtual environment (zsh):

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (from the repo root):

```zsh
pip install -r requirements.txt
```

3) To train models (will attempt to collect data if none are present):

```zsh
python train.py
```

This will:
- collect CF6 data (if `data/` is empty),
- build the feature dataset,
- train decision tree and random forest models (saved to `models/`),
- save feature importance plots to `models/`.

4) To use the prediction API from `weather_predictor.py` (the `predict` function):

- `predict(modeltype, day5, day4, day3, day2, day1)`
  - `modeltype` is either `"besttree"` or `"bestforest"`.
  - Each day argument is a dictionary where keys are 3-letter city codes and values are the CF6 report text (string) for that day.
  - Only `day1` is required for Gradescope-style test cases; the function will accept partial input.

Example (pseudo-code):

```python
from weather_predictor import predict

# day1 is a dict like {"ROC": "<CF6 text>"}
result = predict('besttree', None, None, None, None, day1)
print(result)  # [temp_higher_bool, above_avg_bool, precipitation_bool]
```

Notes & suggestions (proactive review)

- Security / networking:
  - `data_collection.py` disables SSL verification for urllib (ctx.verify_mode = ssl.CERT_NONE). This was helpful during testing but is insecure for public repos. Consider removing or guarding this behavior.

- URL construction and scraping:
  - `construct_cf6_url` currently ignores the `office_code` and does not use `year/month` — this is fine for the scraping strategy used now, but makes the function confusing. Either use all arguments or remove unused params.
  - Scraping logic attempts to find CF6 `pre` tags. The HTML on NWS can change; consider adding better detection or a small cached example input for unit tests.

- Data parsing and robustness:
  - The CF6 parser uses whitespace splitting which works for many reports but can be brittle. Consider adding unit tests for `extract_cf6_data` using a few sample CF6 reports in `tests/`.
  - The code treats trace precipitation `T` as `0.001`. Document this choice in the writeup and/or centralize constants.

- Models & reproducibility:
  - The custom `DecisionTree` and `RandomForest` are pickled and reloaded by `weather_predictor.py`. When sharing, make sure to include the `models.py` source in the repository (it is present).

- Files not to commit (already in .gitignore):
  - `data/`, `daily_data/`, `models/`, `*.pkl`, `debug_*.html`, and Python caches


Acknowledgements

This project uses CF6 reports from the National Weather Service.
# Weather_Prediction
