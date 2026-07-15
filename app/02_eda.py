# STEP 2: Exploratory Data Analysis (EDA) -- Heart Disease


import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

os.makedirs("outputs", exist_ok=True)  # create outputs/ folder if it doesn't exist yet
sns.set_style("whitegrid")
df = pd.read_csv("heart_data.csv")

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

print("=" * 60)
print("SHAPE & OVERVIEW")
print("=" * 60)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df[CONTINUOUS].describe().round(2))

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)
print(df["target"].value_counts())
print(f"Positive class rate: {df['target'].mean():.1%}")
print(">> Close to balanced (unlike diabetes's ~27%). Accuracy is a more")
print("   trustworthy metric here, but recall still matters most --")
print("   missing an at-risk patient is worse than a false alarm.")

print("\n" + "=" * 60)
print("MISSING / IMPOSSIBLE VALUES CHECK")
print("=" * 60)
print("Null counts:")
print(df.isnull().sum().sum(), "total nulls")
# Defensive check: physiologically impossible zeros, same habit as the diabetes project
for col in ["trestbps", "chol", "thalach"]:
    n_zero = (df[col] == 0).sum()
    print(f"  {col:12s}: {n_zero} zero-values (0 is physiologically impossible here)")
print(">> This dataset is clean. Always check anyway -- real clinical exports")
print("   often aren't this well-behaved.")

# --- Continuous feature distributions by outcome ---
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for i, col in enumerate(CONTINUOUS):
    sns.histplot(data=df, x=col, hue="target", kde=True, ax=axes[i],
                 palette=["#4C72B0", "#C44E52"], legend=(i == 0))
    axes[i].set_title(col)
plt.suptitle("Continuous Feature Distributions by Heart Disease Outcome (0=No, 1=Yes)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/01_continuous_distributions.png", dpi=120)
plt.close()

# --- Categorical feature breakdown by outcome (proportion within each category) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, col in enumerate(CATEGORICAL):
    ax = axes[i // 4, i % 4]
    ct = pd.crosstab(df[col], df["target"], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#C44E52"], legend=(i == 0))
    ax.set_title(col)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=0)
plt.suptitle("Categorical Features: Disease Rate by Category", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/02_categorical_breakdown.png", dpi=120)
plt.close()

# --- Correlation heatmap ---
plt.figure(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, annot_kws={"size": 8})
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/03_correlation_heatmap.png", dpi=120)
plt.close()

print("\n" + "=" * 60)
print("TOP CORRELATIONS WITH TARGET")
print("=" * 60)
print(corr["target"].drop("target").sort_values(key=abs, ascending=False).round(3))

print("\nSaved 3 EDA plots to outputs/")
