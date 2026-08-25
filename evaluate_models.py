"""Leakage-aware evaluation for the crop-yield models.

The original notebooks compare models on a 2021-2022 holdout. This script adds
walk-forward model selection, stronger regression metrics, and a previous-year
yield baseline while keeping that holdout untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


NUMERIC_FEATURES = [
    "state_rainfall_val",
    "state_temperature_max_val",
    "state_temperature_min_val",
    "Live Cap FRL",
    "FRL",
    "Level",
    "Current Live Storage",
]
HISTORY_FEATURES = ["yield_lag_1", "yield_trailing_3_mean"]
MODEL_FEATURES = [*NUMERIC_FEATURES, *HISTORY_FEATURES]
GROUP_COLUMNS = ["state_name", "crop_name", "year"]
TARGET = "yield"


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    excluded_states: tuple[str, ...]


DATASETS = {
    "gram": DatasetSpec(
        "merged_gram_reservoir.csv", ("Odisha", "Tamil Nadu")
    ),
    "massor": DatasetSpec(
        "merged_massor_reservoir.csv", ("Odisha", "Telangana")
    ),
    "mustard": DatasetSpec(
        "merged_mustard_reservoir.csv", ("Jharkhand", "Uttarakhand")
    ),
    "potato": DatasetSpec(
        "merged_potato_reservoir.csv",
        ("Andhra Pradesh", "Tamil Nadu", "Telangana", "West Bengal"),
    ),
    "rice": DatasetSpec(
        "merged_rabi_rice_reservoir.csv", ("Jharkhand", "Uttarakhand")
    ),
    "wheat": DatasetSpec(
        "merged_wheat_reservoir.csv", ("Odisha", "Tamil Nadu")
    ),
}


def prepare_annual_data(path: Path, excluded_states: tuple[str, ...]) -> pd.DataFrame:
    """Load one crop dataset and reproduce the notebooks' annual aggregation."""
    raw = pd.read_csv(path)
    required = {
        "temperature_recorded_date",
        "state_name",
        "crop_name",
        TARGET,
        *NUMERIC_FEATURES,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

    raw["temperature_recorded_date"] = pd.to_datetime(
        raw["temperature_recorded_date"], errors="coerce"
    )
    raw["year"] = raw["temperature_recorded_date"].dt.year
    raw = raw[raw["year"].notna() & (raw["year"] < 2023)].copy()
    raw["year"] = raw["year"].astype(int)
    raw = raw[~raw["state_name"].isin(excluded_states)]

    aggregations = {column: "mean" for column in NUMERIC_FEATURES}
    aggregations["state_rainfall_val"] = "sum"
    aggregations[TARGET] = "mean"
    annual = raw.groupby(GROUP_COLUMNS, as_index=False).agg(aggregations)
    annual = annual.dropna(subset=[TARGET]).sort_values(GROUP_COLUMNS)
    if annual.empty:
        raise ValueError(f"{path.name} has no usable annual rows")

    series = annual.groupby(["state_name", "crop_name"], sort=False)[TARGET]
    annual["yield_trailing_3_mean"] = series.transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    previous_year = annual[["state_name", "crop_name", "year", TARGET]].copy()
    previous_year["year"] += 1
    previous_year = previous_year.rename(columns={TARGET: "yield_lag_1"})
    annual = annual.merge(
        previous_year,
        on=["state_name", "crop_name", "year"],
        how="left",
        validate="one_to_one",
    )
    return annual.reset_index(drop=True)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                MODEL_FEATURES,
            ),
            (
                "state",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["state_name"],
            ),
        ],
        remainder="drop",
    )


def _pipeline(estimator: RegressorMixin, *, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [("features", _preprocessor())]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_models() -> dict[str, Callable[[], Pipeline]]:
    """Return fresh model factories so every fold starts from an unfitted model."""
    return {
        "Linear Regression": lambda: _pipeline(LinearRegression()),
        "Random Forest": lambda: _pipeline(
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        ),
        "XGBoost": lambda: _pipeline(
            XGBRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        ),
        "Gradient Boosting": lambda: _pipeline(
            GradientBoostingRegressor(random_state=42)
        ),
        "SVR": lambda: _pipeline(SVR(), scale=True),
        "Voting Ensemble": lambda: _pipeline(
            VotingRegressor(
                estimators=[
                    ("linear", LinearRegression()),
                    (
                        "forest",
                        RandomForestRegressor(
                            n_estimators=300, random_state=42, n_jobs=-1
                        ),
                    ),
                    (
                        "xgboost",
                        XGBRegressor(
                            n_estimators=300, random_state=42, n_jobs=-1
                        ),
                    ),
                ]
            )
        ),
    }


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute scale-dependent and scale-independent regression metrics."""
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mean_abs = float(np.mean(np.abs(y)))
    total_abs = float(np.sum(np.abs(y)))
    return {
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "mae": mae,
        "rmse": rmse,
        "nrmse_pct": 100.0 * rmse / mean_abs if mean_abs else float("nan"),
        "wape_pct": 100.0 * float(np.sum(np.abs(y - pred))) / total_abs
        if total_abs
        else float("nan"),
        "samples": float(len(y)),
    }


def previous_year_baseline(annual: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict each yield from the same state/crop's previous observed year.

    If the immediately previous year is unavailable, use that state/crop's
    historical mean, then the global historical mean as a final fallback.
    """
    lookup = annual.set_index(GROUP_COLUMNS)[TARGET]
    predictions: list[float] = []
    for row in test.itertuples(index=False):
        key = (row.state_name, row.crop_name, row.year - 1)
        if key in lookup.index:
            predictions.append(float(lookup.loc[key]))
            continue

        history = annual[
            (annual["state_name"] == row.state_name)
            & (annual["crop_name"] == row.crop_name)
            & (annual["year"] < row.year)
        ][TARGET]
        if history.empty:
            history = annual[annual["year"] < row.year][TARGET]
        predictions.append(float(history.mean()))
    return np.asarray(predictions)


def walk_forward_results(
    annual: pd.DataFrame,
    model_factories: dict[str, Callable[[], Pipeline]],
    *,
    holdout_start: int,
    cv_years: int,
    min_train_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models on successive years using only earlier years for training."""
    development = annual[annual["year"] < holdout_start]
    years = sorted(development["year"].unique())
    eligible = years[min_train_years:]
    fold_years = eligible[-cv_years:]
    if not fold_years:
        raise ValueError("not enough pre-holdout years for walk-forward validation")

    fold_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    feature_columns = [*MODEL_FEATURES, "state_name"]

    for model_name, factory in model_factories.items():
        all_actual: list[float] = []
        all_predicted: list[float] = []
        for year in fold_years:
            train = development[development["year"] < year]
            validation = development[development["year"] == year]
            if train.empty or validation.empty:
                continue

            model = factory()
            model.fit(train[feature_columns], train[TARGET])
            predicted = model.predict(validation[feature_columns])
            metrics = regression_metrics(validation[TARGET], predicted)
            fold_rows.append({"model": model_name, "year": year, **metrics})
            all_actual.extend(validation[TARGET].astype(float).tolist())
            all_predicted.extend(np.asarray(predicted, dtype=float).tolist())

        if not all_actual:
            continue
        metrics = regression_metrics(pd.Series(all_actual), np.asarray(all_predicted))
        aggregate_rows.append(
            {
                "model": model_name,
                "folds": len(fold_years),
                "first_validation_year": min(fold_years),
                "last_validation_year": max(fold_years),
                **metrics,
            }
        )

    aggregate = pd.DataFrame(aggregate_rows).sort_values("rmse").reset_index(drop=True)
    return pd.DataFrame(fold_rows), aggregate


def evaluate_dataset(
    name: str,
    annual: pd.DataFrame,
    model_factories: dict[str, Callable[[], Pipeline]],
    *,
    holdout_start: int,
    holdout_end: int,
    cv_years: int,
    min_train_years: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a model by walk-forward RMSE and evaluate it once on the holdout."""
    folds, cv = walk_forward_results(
        annual,
        model_factories,
        holdout_start=holdout_start,
        cv_years=cv_years,
        min_train_years=min_train_years,
    )
    selected_name = str(cv.iloc[0]["model"])

    train = annual[annual["year"] < holdout_start]
    holdout = annual[annual["year"].between(holdout_start, holdout_end)].copy()
    if holdout.empty:
        raise ValueError(f"{name} has no {holdout_start}-{holdout_end} holdout rows")

    feature_columns = [*MODEL_FEATURES, "state_name"]
    model = model_factories[selected_name]()
    model.fit(train[feature_columns], train[TARGET])
    predicted = model.predict(holdout[feature_columns])
    baseline = previous_year_baseline(annual, holdout)
    model_metrics = regression_metrics(holdout[TARGET], predicted)
    baseline_metrics = regression_metrics(holdout[TARGET], baseline)
    improvement = (
        100.0 * (baseline_metrics["rmse"] - model_metrics["rmse"])
        / baseline_metrics["rmse"]
        if baseline_metrics["rmse"]
        else float("nan")
    )

    predictions = holdout[[*GROUP_COLUMNS, TARGET]].copy()
    predictions["prediction"] = predicted
    predictions["previous_year_baseline"] = baseline
    summary = {
        "dataset": name,
        "selected_model": selected_name,
        "train_through": holdout_start - 1,
        "holdout_years": f"{holdout_start}-{holdout_end}",
        **{f"holdout_{key}": value for key, value in model_metrics.items()},
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        "rmse_improvement_vs_baseline_pct": improvement,
    }
    cv.insert(0, "dataset", name)
    folds.insert(0, "dataset", name)
    predictions.insert(0, "dataset", name)
    return summary, folds, cv, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../ISI_dataset"),
        help="directory containing merged_*_reservoir.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="directory for CSV evaluation reports",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--holdout-start", type=int, default=2021)
    parser.add_argument("--holdout-end", type=int, default=2022)
    parser.add_argument("--cv-years", type=int, default=5)
    parser.add_argument("--min-train-years", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        args.data_dir / DATASETS[name].filename
        for name in args.datasets
        if not (args.data_dir / DATASETS[name].filename).is_file()
    ]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Missing raw datasets:\n{missing_text}")

    factories = build_models()
    summaries: list[dict[str, object]] = []
    all_folds: list[pd.DataFrame] = []
    all_cv: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for name in args.datasets:
        spec = DATASETS[name]
        annual = prepare_annual_data(args.data_dir / spec.filename, spec.excluded_states)
        summary, folds, cv, predictions = evaluate_dataset(
            name,
            annual,
            factories,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
            cv_years=args.cv_years,
            min_train_years=args.min_train_years,
        )
        summaries.append(summary)
        all_folds.append(folds)
        all_cv.append(cv)
        all_predictions.append(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "holdout_summary.csv", index=False)
    pd.concat(all_cv, ignore_index=True).to_csv(
        args.output_dir / "walk_forward_summary.csv", index=False
    )
    pd.concat(all_folds, ignore_index=True).to_csv(
        args.output_dir / "walk_forward_folds.csv", index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output_dir / "holdout_predictions.csv", index=False
    )

    columns = [
        "dataset",
        "selected_model",
        "holdout_r2",
        "holdout_mae",
        "holdout_rmse",
        "holdout_nrmse_pct",
        "holdout_wape_pct",
        "rmse_improvement_vs_baseline_pct",
    ]
    print(summary_frame[columns].round(4).to_string(index=False))
    print(f"\nReports written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
