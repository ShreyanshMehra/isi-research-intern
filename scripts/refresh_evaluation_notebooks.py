"""Append and execute the shared evaluation section in every crop notebook."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


MARKER = "<!-- leakage-safe-evaluation -->"
NOTEBOOKS = {
    "gram.ipynb": "gram",
    "massor.ipynb": "massor",
    "mustard.ipynb": "mustard",
    "potato.ipynb": "potato",
    "rabi.ipynb": "rice",
    "wheat.ipynb": "wheat",
}


def evaluation_cells(crop: str) -> list[nbformat.NotebookNode]:
    markdown = nbformat.v4.new_markdown_cell(
        f"""{MARKER}
## Leakage-safe evaluation

This section evaluates the **{crop}** yield model as a rolling one-year-ahead
forecast. Models are selected using five-year walk-forward validation on data
through 2020 and evaluated once on the untouched 2021-2022 holdout.

- **Previous-year baseline:** predicts the current yield from the previous
  calendar year's observed yield.
- **Lagged features:** previous-year yield and the trailing three-year mean;
  both use only earlier targets.
- **WAPE:** total absolute error divided by total actual yield. Lower is better.
- **Normalized RMSE:** RMSE divided by mean absolute yield. Lower is better.

Observed weather and reservoir variables are used for the evaluation year, so
these results evaluate the yield-regression stage rather than the complete
Prophet-to-yield forecasting pipeline."""
    )

    setup = nbformat.v4.new_code_cell(
        f"""from pathlib import Path
import os
import sys
from IPython.display import display
import pandas as pd

repo_root = Path.cwd().resolve()
while not (repo_root / "evaluate_models.py").is_file():
    if repo_root.parent == repo_root:
        raise FileNotFoundError("Could not locate evaluate_models.py")
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from evaluate_models import (  # noqa: E402
    DATASETS,
    build_models,
    evaluate_dataset,
    prepare_annual_data,
)

crop_key = "{crop}"
spec = DATASETS[crop_key]
data_dir = Path(os.environ.get("ISI_DATA_DIR", repo_root / "ISI_dataset"))
annual_evaluation_data = prepare_annual_data(
    data_dir / spec.filename, spec.excluded_states
)"""
    )

    evaluate = nbformat.v4.new_code_cell(
        """evaluation_summary, fold_results, cv_results, holdout_predictions = evaluate_dataset(
    crop_key,
    annual_evaluation_data,
    build_models(),
    holdout_start=2021,
    holdout_end=2022,
    cv_years=5,
    min_train_years=5,
)

print("Walk-forward validation (sorted by RMSE):")
display(
    cv_results[["model", "folds", "r2", "mae", "rmse", "nrmse_pct", "wape_pct"]]
    .round(4)
)

print("2021-2022 holdout result:")
display(
    pd.DataFrame([evaluation_summary])[
        [
            "selected_model",
            "holdout_samples",
            "holdout_r2",
            "holdout_mae",
            "holdout_rmse",
            "holdout_nrmse_pct",
            "holdout_wape_pct",
            "baseline_rmse",
            "rmse_improvement_vs_baseline_pct",
        ]
    ].round(4)
)"""
    )
    return [markdown, setup, evaluate]


def refresh_notebook(path: Path, crop: str, repo_root: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    marker_index = next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if MARKER in str(cell.get("source", ""))
        ),
        None,
    )
    if marker_index is not None:
        notebook.cells = notebook.cells[:marker_index]

    cells = evaluation_cells(crop)
    execution_notebook = nbformat.v4.new_notebook(
        cells=[copy.deepcopy(cell) for cell in cells[1:]]
    )
    client = NotebookClient(
        execution_notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
    )
    executed = client.execute()
    cells[1].outputs = executed.cells[0].outputs
    cells[1].execution_count = executed.cells[0].execution_count
    cells[2].outputs = executed.cells[1].outputs
    cells[2].execution_count = executed.cells[1].execution_count
    notebook.cells.extend(cells)
    nbformat.write(notebook, path)
    print(f"updated {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing the merged crop/reservoir CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["ISI_DATA_DIR"] = str(args.data_dir.resolve())
    for filename, crop in NOTEBOOKS.items():
        refresh_notebook(repo_root / "notebooks" / filename, crop, repo_root)


if __name__ == "__main__":
    main()
