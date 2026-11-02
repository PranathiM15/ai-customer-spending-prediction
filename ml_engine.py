"""Data preparation, feature engineering, model training and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_FEATURES = [
    "age", "gender", "annual_income", "purchase_frequency",
    "avg_order_value", "website_visits", "loyalty_years",
    "discount_usage", "total_purchases",
]
NUMERIC_BASE = [
    "age", "annual_income", "purchase_frequency", "avg_order_value",
    "website_visits", "loyalty_years", "discount_usage", "total_purchases",
]
CATEGORICAL_BASE = ["gender"]
ENGINEERED_FEATURES = ["monthly_purchase_value", "visits_per_purchase"]
MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
TARGET = "spending_category"


class DatasetError(ValueError):
    """Raised when an uploaded CSV does not match the app's dataset format."""


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    # Accept common human-readable variants after normalisation.
    aliases = {
        "annual_income_inr": "annual_income",
        "income": "annual_income",
        "purchase_frequency_per_month": "purchase_frequency",
        "avg_order_value_inr": "avg_order_value",
        "average_order_value": "avg_order_value",
        "website_visit": "website_visits",
        "loyalty_year": "loyalty_years",
        "discount_usage_rate": "discount_usage",
        "annual_spending": "annual_spend",
        "spending_level": "spending_category",
        "spending_class": "spending_category",
    }
    return out.rename(columns=aliases)


def load_and_clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean an uploaded dataset; keep annual_spend out of model features."""
    if raw is None or raw.empty:
        raise DatasetError("The uploaded CSV is empty. Add customer rows and try again.")

    df = _normalise_columns(raw)
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise DatasetError(
            "Missing required columns: " + ", ".join(missing)
            + ". Use the sample CSV as the column template."
        )

    original_rows = len(df)
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for col in NUMERIC_BASE + ["annual_spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gender"] = df["gender"].astype("string").str.strip().replace("", pd.NA)

    if TARGET not in df.columns:
        if "annual_spend" not in df.columns:
            raise DatasetError(
                "Add a 'spending_category' column (Low/Medium/High), or provide 'annual_spend' "
                "so the app can create spending categories."
            )
        spend = df["annual_spend"]
        if spend.notna().sum() < 6:
            raise DatasetError("At least 6 valid annual_spend values are needed to create categories.")
        rank = spend.rank(method="first")
        df[TARGET] = pd.qcut(rank, q=3, labels=["Low", "Medium", "High"]).astype("string")
    else:
        aliases = {
            "low": "Low", "low spender": "Low", "low spending": "Low",
            "l": "Low", "medium": "Medium", "mid": "Medium",
            "medium spender": "Medium", "medium spending": "Medium", "m": "Medium",
            "high": "High", "high spender": "High", "high spending": "High", "h": "High",
        }
        labels = df[TARGET].astype("string").str.strip().str.lower()
        df[TARGET] = labels.map(aliases)

    # Keep invalid target rows out; numeric missing values are imputed in the ML pipeline.
    before_target_filter = len(df)
    df = df[df[TARGET].notna()].copy()
    target_rows_removed = before_target_filter - len(df)
    if df.empty:
        raise DatasetError("No valid Low/Medium/High target labels were found.")

    for col in NUMERIC_BASE:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gender"] = df["gender"].fillna("Unknown").astype(str).str.title()
    df[TARGET] = df[TARGET].astype(str)

    if df[NUMERIC_BASE].isna().all(axis=None):
        raise DatasetError("The numeric customer columns contain no usable values.")

    # Add derived fields for the model and for a more informative workflow.
    df = add_engineered_features(df)
    report = {
        "input_rows": original_rows,
        "clean_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "invalid_target_rows_removed": target_rows_removed,
        "missing_numeric_cells": int(df[NUMERIC_BASE].isna().sum().sum()),
        "columns_used": len(BASE_FEATURES),
    }
    return df.reset_index(drop=True), report


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["monthly_purchase_value"] = out["purchase_frequency"] * out["avg_order_value"]
    safe_frequency = out["purchase_frequency"].clip(lower=1)
    out["visits_per_purchase"] = out["website_visits"] / safe_frequency
    return out


def _build_preprocessor() -> ColumnTransformer:
    numeric_features = NUMERIC_BASE + ENGINEERED_FEATURES
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, CATEGORICAL_BASE),
    ])


def _aggregate_importance(pipeline: Pipeline, raw_features: list[str]) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    transformed_names = preprocessor.get_feature_names_out()
    totals = {feature: 0.0 for feature in raw_features}
    for name, value in zip(transformed_names, values):
        clean_name = str(name).split("__", 1)[-1]
        matched = next((f for f in raw_features if clean_name == f or clean_name.startswith(f + "_")), None)
        if matched is not None:
            totals[matched] += float(value)
    result = pd.DataFrame({"feature": list(totals), "importance": list(totals.values())})
    total = result["importance"].sum()
    if total > 0:
        result["importance"] = result["importance"] / total
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train three baseline classifiers and return hold-out metrics and artifacts."""
    X = add_engineered_features(df)[MODEL_FEATURES].copy()
    y = df[TARGET].astype(str).copy()

    counts = y.value_counts()
    if len(counts) < 2:
        raise DatasetError("The target must contain at least two spending categories.")
    if counts.min() < 4:
        raise DatasetError(
            "Each spending category needs at least 4 rows for a reliable train/test split. "
            "Add more examples or use the bundled sample dataset."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1600, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
    }

    fitted = {}
    records = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", _build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        fitted[name] = pipeline
        records.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })

    metrics = pd.DataFrame(records).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["Model"])
    best_pipeline = fitted[best_name]
    importance = _aggregate_importance(best_pipeline, MODEL_FEATURES)
    return {
        "models": fitted,
        "metrics": metrics,
        "best_model_name": best_name,
        "best_model": best_pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": sorted(y.unique().tolist()),
        "feature_importance": importance,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def prepare_customer_input(values: dict) -> pd.DataFrame:
    """Build a one-row feature frame matching the training schema."""
    row = {feature: values.get(feature) for feature in BASE_FEATURES}
    frame = pd.DataFrame([row])
    for col in NUMERIC_BASE:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["gender"] = frame["gender"].fillna("Unknown").astype(str).str.title()
    return add_engineered_features(frame)[MODEL_FEATURES]
