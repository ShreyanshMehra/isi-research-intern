# Model evaluation

The original notebooks use a chronological split: training through 2020 and a
2021-2022 test set. `evaluate_models.py` strengthens that evaluation without
using the holdout for model selection.

It provides:

- walk-forward validation over the final five pre-2021 years;
- model selection by validation RMSE;
- a single final evaluation on the untouched 2021-2022 holdout;
- R2, MAE, RMSE, normalized RMSE, and WAPE;
- comparison with a previous-year yield baseline; and
- per-year, per-crop predictions for error analysis.

The evaluated regressors include two history features that are available at
forecast time: the previous calendar year's yield and the trailing three-year
mean yield. Both are created with shifts, so a row never uses its own target or
future targets as features.

Run it from the repository root after restoring the raw `ISI_dataset` folder:

```powershell
python evaluate_models.py --data-dir ..\ISI_dataset
```

The reports are written to `evaluation_results/`:

- `holdout_summary.csv`: final metrics and improvement over the baseline;
- `walk_forward_summary.csv`: aggregate validation metrics for every model;
- `walk_forward_folds.csv`: metrics for each model and validation year; and
- `holdout_predictions.csv`: actual, model, and baseline values for each row.

Do not describe the 2021-2022 result as an end-to-end forecast score. These
tests use the observed environmental variables for those years and therefore
evaluate the yield-regression stage. Prophet forecasts should be backtested
separately before claiming an end-to-end pipeline metric.
