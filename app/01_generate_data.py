# STEP 1: Generate a clinically-realistic synthetic heart disease dataset.

import numpy as np
import pandas as pd

np.random.seed(7)
N = 900  # number of patients

# --- Demographics ---
age = np.random.normal(54, 9, N).clip(29, 77).astype(int)
sex = np.random.binomial(1, 0.68, N)  # 1 = male, 0 = female; dataset skews male

# Latent, unmeasured cardiovascular risk (genetics, diet, smoking, etc.)
risk_factor = np.random.normal(0, 1, N)

# --- Clinical measurements, correlated with age, sex, and hidden risk ---
trestbps = 120 + 0.5 * (age - 54) + 6 * risk_factor + 4 * sex + np.random.normal(0, 10, N)
trestbps = trestbps.clip(94, 200)

chol = 210 + 1.0 * (age - 54) + 25 * risk_factor + np.random.normal(0, 30, N)
chol = chol.clip(126, 564)

fbs = (np.random.rand(N) < (0.10 + 0.08 * (risk_factor > 1))).astype(int)

# Max heart rate DECREASES with age and with higher cardiac risk
thalach = 202 - 0.8 * age - 12 * risk_factor + np.random.normal(0, 12, N)
thalach = thalach.clip(71, 202)

exang = (np.random.rand(N) < (0.15 + 0.25 * (risk_factor > 0.5))).astype(int)

oldpeak = (0.8 + 0.9 * risk_factor + np.random.exponential(0.4, N)).clip(0, 6.2)

# --- Categorical clinical test results ---
# cp: 0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic
# asymptomatic (3) is actually the highest-risk category clinically
cp_probs_low_risk = [0.45, 0.30, 0.20, 0.05]
cp_probs_high_risk = [0.10, 0.15, 0.15, 0.60]
cp = np.array([
    np.random.choice([0, 1, 2, 3], p=cp_probs_high_risk if rf > 0.5 else cp_probs_low_risk)
    for rf in risk_factor
])

restecg = np.random.choice([0, 1, 2], size=N, p=[0.55, 0.40, 0.05])

slope = np.array([
    np.random.choice([0, 1, 2], p=[0.10, 0.30, 0.60]) if rf > 0.5
    else np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10])
    for rf in risk_factor
])

ca = np.array([
    np.random.choice([0, 1, 2, 3], p=[0.15, 0.25, 0.30, 0.30]) if rf > 0.5
    else np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03])
    for rf in risk_factor
])

thal = np.array([
    np.random.choice([0, 1, 2], p=[0.10, 0.15, 0.75]) if rf > 0.5
    else np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
    for rf in risk_factor
])  # 0=normal, 1=fixed defect, 2=reversible defect

# --- Generate outcome (heart disease presence) from real risk drivers ---
z = (
    -1.9
    + 0.035 * age
    + 0.55 * sex
    + 0.55 * (cp == 3)
    + 0.012 * trestbps
    + 0.006 * chol
    - 0.028 * thalach
    + 0.9 * exang
    + 0.55 * oldpeak
    + 0.35 * ca
    + 0.5 * (thal == 2)
    + 0.7 * risk_factor
)
prob_disease = 1 / (1 + np.exp(-z))
target = np.random.binomial(1, prob_disease)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps.round(1),
    "chol": chol.round(1), "fbs": fbs, "restecg": restecg,
    "thalach": thalach.round(1), "exang": exang, "oldpeak": oldpeak.round(2),
    "slope": slope, "ca": ca, "thal": thal, "target": target,
})

df.to_csv("heart_data.csv", index=False)

print(f"Generated {N} synthetic patient records.")
print(f"Class balance -> Heart disease: {target.sum()} ({target.mean():.1%}) | "
      f"No disease: {N - target.sum()} ({1 - target.mean():.1%})")
print("Saved to heart_data.csv\n")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
