# Evaluation results

Models were selected independently for each crop using five-year walk-forward
validation on data through 2020. The selected model was then evaluated once on
the 2021-2022 holdout. All results are for rolling one-year-ahead predictions.

| Crop | Selected model | Samples | R2 | MAE | RMSE | Previous-year RMSE | RMSE change |
|---|---|---:|---:|---:|---:|---:|---:|
| Gram | SVR | 24 | 0.858 | 0.103 | 0.125 | 0.147 | 15.1% better |
| Massor | Random Forest | 14 | -0.625 | 0.226 | 0.460 | 0.443 | 3.9% worse |
| Mustard | Linear Regression | 16 | 0.960 | 0.087 | 0.112 | 0.127 | 12.1% better |
| Potato | Linear Regression | 8 | 0.755 | 4.135 | 5.197 | 2.742 | 89.5% worse |
| Rice | Gradient Boosting | 6 | 0.874 | 0.230 | 0.287 | 0.249 | 15.3% worse |
| Wheat | Random Forest | 24 | 0.926 | 0.187 | 0.251 | 0.196 | 27.8% worse |

Five of the six crop-specific models achieved holdout R2 above 0.75. However,
only the gram and mustard models beat the previous-year persistence baseline on
RMSE. This baseline comparison prevents a high R2 caused by stable year-to-year
crop yields from being presented as model improvement.

The regressors use observed weather and reservoir variables for the evaluation
year. These figures therefore evaluate the yield-regression stage, not the full
Prophet-to-yield pipeline. Prophet inputs require separate rolling backtests
before an end-to-end forecasting claim is made.
