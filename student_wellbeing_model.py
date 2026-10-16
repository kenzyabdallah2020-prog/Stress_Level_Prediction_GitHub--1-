# =========================
# Import Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("student_wellbeing.csv")

print("\nClass Distribution:")
print(df["stress_level"].value_counts())

print("\nClass Percentage:")
print(df["stress_level"].value_counts(normalize=True) * 100)

numeric_cols = [
    "sleep_hours",
    "study_hours",
    "screen_time",
    "attendance",
    "anxiety_level",
    "academic_pressure",
    "social_support",
    "physical_activity"
]

# =========================
# Visualization Functions
# =========================

def show_stress(data, title):
    counts = data["stress_level"].value_counts()

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.show()


def show_missing(data, title):
    missing = data.isnull().sum()

    plt.figure()
    plt.bar(missing.index, missing.values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def show_box(data, title):
    plt.figure()

    plt.boxplot(
        [data[col].dropna() for col in numeric_cols],
        labels=numeric_cols
    )

    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# =========================
# Before Cleaning
# =========================

print("==== BEFORE CLEANING ====")

print("Missing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

show_stress(df, "Stress Before Cleaning")
show_missing(df, "Missing Before Cleaning")
show_box(df, "Outliers Before Cleaning")

# =========================
# Data Cleaning
# =========================

df = df.drop_duplicates()

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["stress_level"] = df["stress_level"].fillna(
    df["stress_level"].mode()[0]
)

df["sleep_hours"] = df["sleep_hours"].clip(3, 10)
df["study_hours"] = df["study_hours"].clip(0, 10)
df["screen_time"] = df["screen_time"].clip(1, 12)
df["attendance"] = df["attendance"].clip(30, 100)

df["anxiety_level"] = df["anxiety_level"].clip(1, 10)
df["academic_pressure"] = df["academic_pressure"].clip(1, 10)
df["social_support"] = df["social_support"].clip(1, 10)
df["physical_activity"] = df["physical_activity"].clip(0, 10)

# =========================
# After Cleaning
# =========================

print("\n==== AFTER CLEANING ====")

print("Missing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

show_stress(df, "Stress After Cleaning")
show_missing(df, "Missing After Cleaning")
show_box(df, "Outliers After Cleaning")

# =========================
# Feature Engineering
# =========================

df["screen_study_ratio"] = (
    df["screen_time"] / (df["study_hours"] + 1)
)

df["pressure_index"] = (
    df["anxiety_level"] + df["academic_pressure"]
)

df["health_score"] = (
    df["sleep_hours"] + df["physical_activity"]
)

df["support_attendance_score"] = (
    df["social_support"] + (df["attendance"] / 10)
)

print("\nFeatures Added Successfully")

# =========================
# Encoding
# =========================

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Grid Search - Random Forest
# =========================

param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest RF Params:", grid_search.best_params_)
print("Best CV Accuracy:", round(grid_search.best_score_ * 100, 2), "%")

# =========================
# Models
# =========================

rf_model = grid_search.best_estimator_

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

svm_model = SVC(
    kernel="rbf",
    class_weight="balanced",
    random_state=42
)

# =========================
# Training & Evaluation
# =========================

models = {
    "Random Forest": (
        rf_model,
        X_train,
        X_test
    ),

    "Logistic Regression": (
        lr_model,
        X_train_scaled,
        X_test_scaled
    ),

    "SVM": (
        svm_model,
        X_train_scaled,
        X_test_scaled
    )
}

results = {}

for name, (model, X_tr, X_te) in models.items():

    model.fit(X_tr, y_train)

    pred = model.predict(X_te)

    acc = accuracy_score(y_test, pred)

    results[name] = {
        "model": model,
        "pred": pred,
        "acc": acc
    }

    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")

    print(f"Accuracy: {round(acc * 100, 2)}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            pred,
            target_names=encoder.classes_
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

# =========================
# Save Model
# =========================

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel Saved Successfully!")