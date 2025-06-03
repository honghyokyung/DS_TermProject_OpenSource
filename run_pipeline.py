import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

def run_pipeline(df, target_col):
    df = df.copy()
    df.dropna(axis=0, how='all', inplace=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype(str)

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

    pipeline = Pipeline([
        ('preprocessor', full_pipeline),
        ('classifier', RandomForestClassifier())
    ])

    param_grid = {
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
        'classifier__n_estimators': [100, 200]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=2,
        scoring=make_scorer(accuracy_score),
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X, y)
    return {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }
