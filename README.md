# Tempest FWI Predictor

A machine learning powered Flask web application that predicts Fire Weather Index (FWI) from meteorological inputs.

## Project Outcome
- Train and validate a Ridge Regression model for FWI prediction.
- Normalize input features using StandardScaler.
- Serve live predictions through a Flask web interface.
- Package reproducible artifacts for review and submission.

## Repository Structure
- `app.py`: Flask application for live inference.
- `run_pipeline_validation.py`: End-to-end training and artifact generation script.
- `data/`: Source dataset.
- `models/`: Saved model, scaler, and feature schema.
- `artifacts/`: Cleaned dataset and model metrics.
- `templates/`: Frontend pages for input and prediction output.
- `static/`: CSS styling for the web app.
- `docs/`: Submission documentation, diagrams, and screenshots.

## How To Run
1. Activate virtual environment.
2. Install dependencies from your environment specification.
3. Start the Flask app:

```bash
python app.py
```

4. Open http://127.0.0.1:5000 and submit feature values.

## How To Rebuild Training Artifacts
Run:

```bash
python run_pipeline_validation.py
```

This regenerates:
- `models/ridge.pkl`
- `models/scaler.pkl`
- `models/features.json`
- `artifacts/cleaned_fwi_dataset.csv`
- `artifacts/model_metrics.csv`

## Model Evaluation
Evaluation artifacts are saved in `artifacts/model_metrics.csv` using:
- MAE
- RMSE
- R2

## Submission Documents
See:
- `docs/PROJECT_DOCUMENTATION.md`
- `docs/diagrams/system_workflow.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `docs/screenshots/`
