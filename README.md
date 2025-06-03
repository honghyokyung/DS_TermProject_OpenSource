# DS Term Project – Open Source Contribution: `run_pipeline`

This repository contains a single top-level function that performs:
- Automatic preprocessing for numerical and categorical columns
- Classification using `RandomForestClassifier`
- Hyperparameter tuning via `GridSearchCV`
- Automatic selection of the best model configuration

See `example_pipeline.ipynb` for usage demonstration.

---

## Features
- Automatically identifies and preprocesses numerical and categorical features
- Applies scaling (Standard or MinMax) and OneHot encoding
- Trains a classification model using `RandomForestClassifier`
- Uses `GridSearchCV` to find the best preprocessing-model combination

---

## Usage Example

```python
from run_pipeline_with_search import run_pipeline_with_search
import pandas as pd

# Load example dataset
df = pd.read_excel("final_dataset.xlsx")

# Create binary target column
df["is_high_spender"] = df["가계지출금액"] > df["가계지출금액"].median()

# Run pipeline
result = run_pipeline_with_search(df, target_col="is_high_spender")

# Show results
print("Best accuracy:", result["best_score"])
print("Best parameters:", result["best_params"])
