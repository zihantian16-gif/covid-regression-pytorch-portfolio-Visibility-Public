# COVID-19 Trends Regression with PyTorch MLP

An educational, reproducible regression case study adapted from my course-work notebook. It demonstrates data validation, feature scaling fitted on training data only, mean and ridge-regression baselines, a PyTorch multilayer perceptron, early stopping, and evaluation on a held-out random validation split. It predicts a **continuous `tested_positive` value** in the supplied dataset; it is **not** a medical diagnosis tool or an individual infection classifier.

## Data and attribution

The dataset comes from the public [ML2021 Spring HW01: Regression](https://github.com/ga642381/ML2021-Spring/tree/main/HW01) coursework. The code in this folder is a new, condensed portfolio implementation based on the accompanying personal notebook. This is an **educational course project, not paid client work**. Review the dataset's distribution terms before republishing the CSV files; this folder intentionally contains no data copies.

## Run

Use Python 3.10+ and install dependencies with `pip install -r requirements.txt`. Then run:

```bash
python train.py --train /path/to/covid.train.csv --test /path/to/covid.test.csv
```

Outputs are written to `outputs/`: `metrics.json`, `training_history.csv`, `training_curve.png`, `model_comparison.png`, `test_predictions.csv`, and `model.pt`.

The supplied test data has no target column, so its predictions **cannot** be scored locally. Report only the validation metrics from `metrics.json`; the random split is a useful demonstration, not evidence of real-world clinical performance. A stronger follow-up would compare multiple seeds and alternative validation strategies.

## Verified run (seed 42)

The supplied 2,700 labeled rows were split into 2,160 training and 540 validation rows. All 93 input features exclude the `id` column. On this single validation split, RMSE was **7.858** for the mean baseline, **0.977** for ridge regression, and **0.983** for the PyTorch MLP. The MLP's best checkpoint was epoch 124. Ridge is slightly better here, illustrating that a more complex model is not automatically the best choice.
