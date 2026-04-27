# Insurance Cost Predictor

Streamlit application for exploring the Kaggle medical insurance dataset and predicting annual insurance charges from a user profile.

## What the app includes

- `Live Cost Predictor`
  - Profile form for age, sex, BMI, children, region, and smoker status
  - Stratified Block 2 prediction flow
  - Uncertainty interval from quantile regression
  - Local feature-driver visualization
- `Data Exploration`
  - Filterable EDA dashboard
  - K-means validation of the smoker / non-smoker split
- `Model Comparison`
  - Held-out metrics for baseline, tree, neural, quantile, and MDN models
  - MDN from-scratch results

## Project structure

- `app/`
  - Streamlit entrypoint and page modules
- `data/`
  - Source insurance dataset
- `evaluation/`
  - Evaluation helpers
- `exploration/`
  - EDA and clustering utilities
- `models/`
  - Training scripts for regression, uncertainty, and MDN models
- `preprocess/`
  - Shared feature engineering and train/test split logic
- `saved_models/`
  - Saved metrics and trained model artifacts

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Presentation demo order

1. Open `Live Cost Predictor` and show one low-risk, one high-risk, and one unknown-smoker profile.
2. Move to `Data Exploration` to show the smoker/non-smoker separation and clustering evidence.
3. Finish on `Model Comparison` to explain why the final prediction flow uses a stratified approach plus uncertainty.
