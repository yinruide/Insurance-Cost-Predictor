# Usage Guide of 'preprocess.py'
### Author: Ruide Yin

## What It Does

Loads `insurance.csv`, encodes categoricals, engineers features, splits into train/test, and optionally scales. You call two functions and get arrays ready for modeling.

Key design decisions handled automatically:
- **Encoding**: `_linear` / `_torch` / `_mdn` functions use **one-hot** encoding (region → 3 dummy columns); `_tree` functions use **label encoding** (region → single integer). Choose the getter that matches your model type.
- **Column dropping**: `_build_feature_matrix` automatically removes `charges`, `log_charges`, and `smoker` from features in regression tasks. For classification on `smoker`, `bmi_smoker` is also auto-dropped to prevent data leakage. You don't need to drop anything manually.
- **`log_charges`**: Generated as `np.log1p(charges)`. To convert predictions back to original dollar scale, use `np.expm1(pred)`.
- **`get_regressor_data_mdn`** is a thin wrapper around `get_regressor_data_torch` — they return identical DataFrames.

## Import
```python
from preprocess import get_regressor_data_linear, split_scaled  # swap in whichever you need, find details in snippets
```

## Quick Reference

### Block 1 — Regression

| Model | Step 1 | Step 2 | Target | Encoding |
|---|---|---|---|---|
| Linear Regression | `get_regressor_data_linear()` | `split_scaled(df, "log_charges")` | `log_charges` | one-hot |
| Quantile Regression | `get_regressor_data_linear()` | `split_scaled(df, "log_charges")` | `log_charges` | one-hot |
| Random Forest | `get_regressor_data_tree()` | `split_unscaled(df, "charges")` | `charges` | label |
| XGBoost | `get_regressor_data_tree()` | `split_unscaled(df, "charges")` | `charges` | label |
| MLP | `get_regressor_data_torch()` | `split_scaled(df, "charges")` | `charges` | one-hot |
| MDN | `get_regressor_data_mdn()` | `split_scaled(df, "charges")` | `charges` | one-hot |

### Block 2 — Classification (target: `smoker`)

| Model | Step 1 | Step 2 | Encoding |
|---|---|---|---|
| Logistic Regression | `get_classifier_data_logistic()` | `split_scaled(df, "smoker")` | one-hot |
| RF / XGB Classifier | `get_classifier_data_tree()` | `split_unscaled(df, "smoker")` | label |
| MLP Classifier | `get_classifier_data_torch()` | `split_scaled(df, "smoker")` | one-hot |

### Block 2 — Stratified Subgroup Regression

Manually filter by smoker status before splitting:
```python
df = get_regressor_data_torch()

# Smoker subgroup
s_df = df[df["smoker"] == 1]
X_tr, X_te, y_tr, y_te, scaler = split_scaled(s_df, "charges")

# Non-smoker subgroup
n_df = df[df["smoker"] == 0]
X_tr, X_te, y_tr, y_te, scaler = split_scaled(n_df, "charges")
```

> **Note:** After filtering to a single smoker group, stratification is automatically skipped (`nunique() <= 1`), so no error will be raised.

## Feature Columns

After preprocessing, the actual feature columns used in `X` are:

| Encoding | Regression features | Classification features |
|---|---|---|
| one-hot | `age, sex, bmi, children, bmi_smoker, region_northwest, region_southeast, region_southwest` | `age, sex, bmi, children, region_northwest, region_southeast, region_southwest` |
| label | `age, sex, bmi, children, bmi_smoker, region` | `age, sex, bmi, children, region` |

`smoker` is excluded from regression features (used only for stratified splitting).
`bmi_smoker` is excluded from classification features (would cause data leakage).

## Copy-Paste Snippets

Find your model, copy the block, and you're ready to go.

### Block 1 — Regression

**Linear Regression**
```python
from preprocess import get_regressor_data_linear, split_scaled

df = get_regressor_data_linear()
X_train, X_test, y_train, y_test, scaler = split_scaled(df, "log_charges")
# To recover original charges: np.expm1(pred)
```

**Quantile Regression**
```python
from preprocess import get_regressor_data_linear, split_scaled

df = get_regressor_data_linear()
X_train, X_test, y_train, y_test, scaler = split_scaled(df, "log_charges")
# To recover original charges: np.expm1(pred)
```

**Random Forest (Regressor)**
```python
from preprocess import get_regressor_data_tree, split_unscaled

df = get_regressor_data_tree()
X_train, X_test, y_train, y_test = split_unscaled(df, "charges")
```

**XGBoost (Regressor)**
```python
from preprocess import get_regressor_data_tree, split_unscaled

df = get_regressor_data_tree()
X_train, X_test, y_train, y_test = split_unscaled(df, "charges")
```

**MLP (Regressor)**
```python
from preprocess import get_regressor_data_torch, split_scaled

df = get_regressor_data_torch()
X_train, X_test, y_train, y_test, scaler = split_scaled(df, "charges")
```

**MDN**
```python
from preprocess import get_regressor_data_mdn, split_scaled

df = get_regressor_data_mdn()
X_train, X_test, y_train, y_test, scaler = split_scaled(df, "charges")
```

### Block 2 — Classification

**Logistic Regression (Classifier)**
```python
from preprocess import get_classifier_data_logistic, split_scaled

df = get_classifier_data_logistic()
X_train, X_test, y_train, y_test, scaler = split_scaled(df, "smoker")
```

**RF / XGB (Classifier)**
```python
from preprocess import get_classifier_data_tree, split_unscaled

df = get_classifier_data_tree()
X_train, X_test, y_train, y_test = split_unscaled(df, "smoker")
```

**MLP (Classifier)**
```python
from preprocess import get_classifier_data_torch, split_scaled

df = get_classifier_data_torch()
X_train, X_test, y_train, y_test, scaler = split_scaled(df, "smoker")
```

### Block 2 — Stratified Subgroup Regression

**Smoker Subgroup Regressor**
```python
from preprocess import get_regressor_data_torch, split_scaled

df = get_regressor_data_torch()
s_df = df[df["smoker"] == 1]
X_train, X_test, y_train, y_test, scaler = split_scaled(s_df, "charges")
```

**Non-Smoker Subgroup Regressor**
```python
from preprocess import get_regressor_data_torch, split_scaled

df = get_regressor_data_torch()
n_df = df[df["smoker"] == 0]
X_train, X_test, y_train, y_test, scaler = split_scaled(n_df, "charges")
```

## Return Values

**`split_scaled`** returns 5 values:
```python
X_train, X_test, y_train, y_test, scaler = split_scaled(df, target_col)
```

All numpy arrays except `scaler` (a fitted `StandardScaler` — keep it for inference).

**`split_unscaled`** returns 4 values:
```python
X_train, X_test, y_train, y_test = split_unscaled(df, target_col)
```

All numpy arrays. No scaler needed for tree models.

## Notes

- **Path:** All `get_*` functions default to `path="../data/insurance.csv"`. If your working directory is different, pass the correct path explicitly.
- **Same split everywhere:** All splits use `random_state=12138`, `test_size=0.2`, stratified by smoker. Every model sees the same train/test rows.
- **Scaler:** If your model uses `split_scaled`, save the returned `scaler` — you'll need it to transform new user inputs at inference time (Streamlit page 2).
- **Smoker is not a feature for regressors.** It's automatically dropped from X in regression tasks (only used for stratified splitting). Don't add it back manually.
- **`bmi_smoker` is auto-excluded for classifiers.** Since the target is `smoker`, keeping `bmi * smoker` would be data leakage. This is already handled — no action needed.
