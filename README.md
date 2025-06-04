# DS Term Project – Open Source Contribution: `run_pipeline`

This repository provides a single top-level function that performs:
- Automatic preprocessing of numerical and categorical features
- Classification using `DecisionTreeClassifier`
- Hyperparameter tuning via `GridSearchCV`
- Automatic selection of the best model configuration

See `example_pipeline.ipynb` for a complete usage example.

---

## Features
- Automatically detects and preprocesses numerical and categorical columns
- Supports scaling (`StandardScaler`, `MinMaxScaler`) and one-hot encoding
- Trains a `DecisionTreeClassifier` with hyperparameter tuning
- Uses `GridSearchCV` to explore combinations and select the best one
- Returns best score, parameters, estimator, and cross-validation results

---

## Usage Example

```python
from run_pipeline import run_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load example dataset
df = pd.read_excel("final_dataset.xlsx")
df["is_high_spender"] = df["가계지출금액"] > df["가계지출금액"].median()

# Define parameter grid for Decision Tree
param_grid = {
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [3, 5, 7, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 3]
}

# Run pipeline with param_grid
result = run_pipeline(df, target_col="is_high_spender", param_grid=param_grid)

print("Best accuracy:", result["best_score"])
print("Best parameters:", result["best_params"])

