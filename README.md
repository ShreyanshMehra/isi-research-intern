# Crop Yield Forecasting

Crop-specific yield forecasting using more than 20 years of rainfall,
temperature, reservoir, and agricultural production data across Indian states.
The project compares regression models for yield estimation and uses Prophet to
forecast the environmental inputs required for 2023 predictions.

## Pipeline

1. Aggregate weather, reservoir, and yield observations by state, crop, and year.
2. Remove states with insufficient historical coverage for each crop.
3. Engineer state indicators, previous-year yield, and trailing three-year mean yield.
4. Compare Linear Regression, Random Forest, XGBoost, Gradient Boosting, SVR,
   and a Voting Ensemble.
5. Select each crop's model using five-year walk-forward validation through 2020.
6. Evaluate once on the untouched 2021-2022 holdout.
7. Forecast 2023 environmental variables with Prophet and estimate crop yield.

## Evaluation

The chronological evaluation avoids random train/test splitting. Every
validation year is predicted using only earlier years. The previous-year yield
and trailing three-year average are shifted before use, preventing target
leakage.

| Crop | Selected model | Samples | R2 | MAE | RMSE | Previous-year RMSE | RMSE change |
|---|---|---:|---:|---:|---:|---:|---:|
| Gram | SVR | 24 | 0.858 | 0.103 | 0.125 | 0.147 | 15.1% better |
| Massor | Random Forest | 14 | -0.625 | 0.226 | 0.460 | 0.443 | 3.9% worse |
| Mustard | Linear Regression | 16 | 0.960 | 0.087 | 0.112 | 0.127 | 12.1% better |
| Potato | Linear Regression | 8 | 0.755 | 4.135 | 5.197 | 2.742 | 89.5% worse |
| Rice | Gradient Boosting | 6 | 0.874 | 0.230 | 0.287 | 0.249 | 15.3% worse |
| Wheat | Random Forest | 24 | 0.926 | 0.187 | 0.251 | 0.196 | 27.8% worse |

Five of six crop-specific models achieved holdout R2 above 0.75. Gram and
mustard also improved RMSE over the previous-year persistence baseline by
15.1% and 12.1%, respectively. The baseline remains stronger for the other
crops, showing that high R2 alone does not establish forecasting improvement.

### Metrics

- **R2:** proportion of variation explained by the predictions; higher is better.
- **MAE:** average absolute prediction error in the yield's original units.
- **RMSE:** error metric that penalizes larger mistakes more heavily.
- **Normalized RMSE:** RMSE as a percentage of mean absolute yield.
- **WAPE:** total absolute error divided by total actual yield.
- **Previous-year baseline:** predicts each state/crop from its previous year's yield.

## Reproduce the evaluation

Place the six merged CSV files under `ISI_dataset/`, then run:

```powershell
pip install -r requirements.txt
python evaluate_models.py --data-dir .\ISI_dataset
python scripts\refresh_evaluation_notebooks.py --data-dir .\ISI_dataset
```

The evaluator writes detailed reports to `evaluation_results/`. Each crop
notebook also contains an executed **Leakage-safe evaluation** section.

## Important limitation

The reported holdout results use observed weather and reservoir inputs for the
evaluation year. They evaluate the yield-regression stage, not the complete
Prophet-to-yield chain. Prophet inputs must be rolling-backtested before claiming
an end-to-end forecasting score.

See [EVALUATION.md](EVALUATION.md) for the procedure and
[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for the complete interpretation.
