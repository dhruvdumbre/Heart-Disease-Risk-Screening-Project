# STEP 3: Preprocessing, Training, and Evaluation -- Heart Disease

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

os.makedirs("outputs", exist_ok=True)  # create outputs/ folder if it doesn't exist yet
sns.set_style("whitegrid")
OUT = "outputs"

df = pd.read_csv("heart_data.csv")

# --- 1. One-hot encode NOMINAL categorical features ---
# sex, fbs, exang are already binary (0/1) -- no encoding needed.
# ca is a genuine count (0-3 vessels) -- ordinal, kept numeric.
# cp, restecg, slope, thal are category LABELS -- one-hot encode these.
NOMINAL_COLS = ["cp", "restecg", "slope", "thal"]
df_encoded = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)

print(f"Columns before encoding: {df.shape[1]}")
print(f"Columns after encoding : {df_encoded.shape[1]}")
print(f"New columns added: {[c for c in df_encoded.columns if c not in df.columns]}")

X = df_encoded.drop(columns=["target"])
y = df_encoded["target"]

# --- 2. Stratified train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train positive rate: {y_train.mean():.1%} | Test positive rate: {y_test.mean():.1%}")

# --- 3. Scale continuous features (fit on train only -- avoid leakage) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train models ---
log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)  # tree models don't need scaling or one-hot, but reusing for consistency

models = {"Logistic Regression": (log_reg, X_test_scaled), "Random Forest": (rf, X_test)}

# --- 5. Evaluate ---
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

results = []
plt.figure(figsize=(6, 6))
for name, (model, X_te) in models.items():
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1": f1, "ROC-AUC": auc})

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of predicted 'at risk', how many really are)")
    print(f"Recall   : {rec:.3f}  (of actual disease cases, how many we caught)")
    print(f"F1 Score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_roc_curves.png", dpi=120)
plt.close()

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(results_df.round(3).to_string(index=False))
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)

# --- 6. Cross-validation ---
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION (Random Forest, ROC-AUC)")
print("=" * 60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")
print(f"Fold scores: {np.round(cv_scores, 3)}")
print(f"Mean ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# --- 7. Explainability ---
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

coef_df = pd.DataFrame({
    "Feature": X.columns, "Coefficient": log_reg.coef_[0]
}).sort_values("Coefficient")
axes[0].barh(coef_df["Feature"], coef_df["Coefficient"],
             color=["#C44E52" if c < 0 else "#4C72B0" for c in coef_df["Coefficient"]])
axes[0].set_title("Logistic Regression Coefficients\n(positive = raises heart disease risk)")
axes[0].axvline(0, color="black", linewidth=0.8)

imp_df = pd.DataFrame({
    "Feature": X.columns, "Importance": rf.feature_importances_
}).sort_values("Importance")
axes[1].barh(imp_df["Feature"], imp_df["Importance"], color="#55A868")
axes[1].set_title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig(f"{OUT}/05_feature_importance.png", dpi=120)
plt.close()

print("\nTop risk drivers (Random Forest):")
print(imp_df.sort_values("Importance", ascending=False).head(8).round(3).to_string(index=False))

# --- 8. Save model artifacts ---
joblib.dump(rf, "model_random_forest.joblib")
joblib.dump(log_reg, "model_logistic_regression.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")
joblib.dump(NOMINAL_COLS, "nominal_cols.joblib")

print("\nSaved models, scaler, and column info for reuse in predict_new_patient.py")
