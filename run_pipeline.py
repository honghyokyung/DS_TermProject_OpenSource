import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

def run_pipeline(df, target_col, param_grid=None):
    df = df.copy()
    df.dropna(axis=0, how='all', inplace=True)

# Split into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

# Identify numerical and categorical features
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype(str)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) 
    ])
 # Define preprocessing
    full_pipeline = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])
  # Define full pipeline
    pipeline = Pipeline([
        ('preprocessor', full_pipeline),
        ('classifier', DecisionTreeClassifier())
    ])
  # Default param grid
    param_grid = {
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [3, 5, 7, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 3]
    }
  # Run grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=make_scorer(accuracy_score),
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # Extract top 5 combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.sort_values(by='mean_test_score', ascending=False).head(5)

    return {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'top_5_combinations': top_5[['params', 'mean_test_score']]
    }
