# Heart Disease Risk Prediction — Supervised ML Pipeline + Streamlit Demo

A complete, beginner-friendly walkthrough of building a heart disease
risk-screening model from raw clinical data to a demo-able web app.

## Files

| File | What it does |
|---|---|
| `01_generate_data.py` | Builds `heart_data.csv` — clinical patient records |
| `02_eda.py` | Exploratory analysis: distributions, correlations, class balance |
| `03_train_evaluate.py` | Preprocessing, one-hot encoding, training, evaluation — saves model files |
| `04_predict_new_patient.py` | Command-line demo: score a new patient |
| `app.py` | Streamlit web app — form in, risk score out |
| `requirements.txt` | Python packages needed to run `app.py` |

## Run order (from a terminal, in this folder)

```
pip install -r requirements.txt
python 01_generate_data.py
python 02_eda.py
python 03_train_evaluate.py
python 04_predict_new_patient.py
streamlit run app.py
```

Each script depends on the one before it — `03_train_evaluate.py` in
particular must run successfully before `04_predict_new_patient.py` or
`app.py` will work, since it's what saves the `.joblib` model files
(`model_random_forest.joblib`, `model_logistic_regression.joblib`,
`scaler.joblib`, `feature_columns.joblib`, `nominal_cols.joblib`).

**Important:** always regenerate these `.joblib` files locally by running
`03_train_evaluate.py` yourself, rather than copying them from elsewhere —
pickled models are tied to the exact scikit-learn version that created
them, and mismatched versions throw errors like
`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`.

## What each step teaches you

1. **Data**: structure of clinical tabular data — continuous measurements
   (age, blood pressure, cholesterol, max heart rate, ST depression) mixed
   with categorical test results (chest pain type, ECG, thalassemia)
2. **EDA**: checking class balance, confirming feature-outcome relationships
   make clinical sense (e.g. lower max heart rate = higher risk)
3. **Preprocessing**: one-hot encoding nominal categories (`cp`, `restecg`,
   `slope`, `thal`) so Logistic Regression doesn't wrongly treat them as
   ordinal; stratified train/test split; feature scaling
4. **Training & evaluation**: Logistic Regression vs Random Forest,
   precision/recall/F1/ROC-AUC, 5-fold cross-validation, feature importance
5. **Prediction**: turning a trained model into a reusable scoring function
6. **Deployment**: wrapping that function in a Streamlit form so anyone can
   use it without touching code

## Results on this run

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 77.8% | 77.0% | 77.0% | 77.0% | 0.863 |
| Random Forest | 76.1% | 78.9% | 69.0% | 73.6% | 0.864 |

- **5-fold CV ROC-AUC**: 0.879 (±0.020) — consistent across folds
- **Top risk drivers**: `oldpeak` (ST depression) and `thalach` (max heart
  rate) dominate — both stress-test results, not simple vitals
- Logistic Regression edges out Random Forest on recall — worth remembering
  that "more complex" doesn't always mean "better" on smaller tabular data,
  and recall matters most for screening (missing a true case is costlier
  than a false alarm)

## ⚠️ About the data

This uses a **synthetic dataset**, built to mirror the real **UCI Heart
Disease (Cleveland) dataset**: same 13 features, same rough value ranges,
similar near-balanced class distribution (~48% positive vs the real
dataset's ~46%). Generated locally because this environment had no
internet access to download the real file.

### Using real data instead

Download the real UCI Heart Disease (Cleveland) dataset and save it as
`heart_data.csv` in this folder with these exact column names:

```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal, target
```

Everything downstream (`02_eda.py` onward) works unchanged.

## Troubleshooting

- **`FileNotFoundError` on `.joblib` files**: run `03_train_evaluate.py`
  in the same folder as `app.py` first.
- **`AttributeError: ... has no attribute 'multi_class'`** (or similar):
  scikit-learn version mismatch — delete the `.joblib` files and
  regenerate them by running `03_train_evaluate.py` locally.
- **Streamlit shows "missing ScriptRunContext" warnings**: you ran `app.py`
  directly (e.g. inside Jupyter) instead of with `streamlit run app.py`
  from a terminal.
- **A `.py` file throws `NameError: name 'null' is not defined`**: the file
  got saved as raw Jupyter notebook JSON instead of plain Python — re-download
  it and save it directly, without opening/re-saving through Jupyter first.
- **Downloaded a file and it looks wrong**: sanity-check with
  `more filename.py` (Windows) before running it — you should see actual
  Python code, not `"cell_type"` or `"execution_count"` anywhere.

## Next steps toward a real product

1. **Improve the model**: hyperparameter tuning (`GridSearchCV`), try
   XGBoost/LightGBM, or add SHAP for per-patient (not just global)
   explanations
2. **Get real data**: the real UCI dataset proves the pipeline works;
   a real product needs either a licensed clinical dataset or a
   partnership with a clinic willing to share anonymized, consented data
3. **Reframe as decision support, not diagnosis**: "diagnostic" claims
   trigger FDA Software-as-a-Medical-Device regulation in the US — "risk
   screening to prioritize who a doctor looks at next" is safer and still
   valuable for an early-stage product
4. **Talk to clinicians early**: which features are actually available at
   the point of care, and would this fit into a workflow they'd trust?