# Project Documentation

## 1. Project Summary
This project predicts Fire Weather Index (FWI) using a Ridge Regression model trained on Algerian forest fire weather data. It includes preprocessing, feature scaling, model training, metric evaluation, and Flask deployment for real-time prediction.

## 2. End-to-End Pipeline
1. Raw CSV ingestion from `data/Algerian_forest_fires_dataset_UPDATE.csv`.
2. Data cleaning and type conversion.
3. Missing-value handling and region encoding.
4. Feature selection and train/test split.
5. StandardScaler fitting and persistence.
6. Ridge training with alpha tuning.
7. Evaluation using MAE, RMSE, and R2.
8. Artifact persistence in `models/` and `artifacts/`.
9. Flask runtime loads scaler/model and serves predictions.

## 3. Feature Inputs Used In Deployment
Features are loaded dynamically from `models/features.json`. The web form is generated from this schema to keep feature order and naming aligned with training.

## 4. Flask Application Architecture
- Route `/`: renders input form.
- Route `/predict`: collects form values, scales them, predicts FWI, and renders output.
- Runtime model loading from:
  - `models/ridge.pkl`
  - `models/scaler.pkl`

## 5. Evaluation and Optimization Notes
- Model family: Ridge Regression.
- Hyperparameter tuning: alpha grid search.
- Reported metrics stored in `artifacts/model_metrics.csv`.

## 6. Reproducibility
To reproduce training artifacts:

```bash
python run_pipeline_validation.py
```

To run inference app:

```bash
python app.py
```

## 7. Submission Packaging
This repository includes:
- Source code
- Training and inference artifacts
- Evaluation outputs
- Architecture and workflow diagrams
- UI screenshots
- Completion checklist
