import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

## QUESTION 1 ##
# Read the data
df = pd.read_csv("data/project_1_data.csv")

## QUESTION 2 ##
# Statistical analysis
print("\n---- QUESTION 2: DATA ANALYSIS ----\n")
print("Shape of the Data (Rows, Columns):")
print(df.shape, "\n")
print("First Five Rows of Data:")
print(df.head(), "\n")
print("Statistical Summary:")
print(df.describe(), "\n")

# Step-wise averages for X, Y, Z
step_means = df.groupby("Step")[["X", "Y", "Z"]].mean()

# Visualization of distributions and averages for col in ["X", "Y", "Z"]:
for col in ["X", "Y", "Z"]:
    plt.figure(figsize=(10, 4))
    # Left: overall distribution
    plt.subplot(1, 2, 1)
    plt.hist(df[col], bins=25, color="lightblue", edgecolor="black")
    plt.title(f"{col} Distribution")
    plt.xlabel(f"{col} Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)

    # Right: average per step
    plt.subplot(1, 2, 2)
    plt.bar(step_means.index.astype(str), step_means[col], color="lightgreen", edgecolor="black")
    plt.title(f"Average {col} per Step")
    plt.xlabel("Step")
    plt.ylabel(f"{col} Mean")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

# Step distribution with x-axis labels for each step
plt.figure(figsize=(6, 4))
# Count values per step
step_counts = df["Step"].value_counts().sort_index()
# Plot as bar chart
plt.bar(step_counts.index, step_counts.values, color="lightblue", edgecolor="black")
# Ensure x-axis has all steps labeled
plt.xticks(step_counts.index)
plt.title("Step Distribution")
plt.xlabel("Step")
plt.ylabel("Frequency")
plt.grid(alpha=0.3, axis="y")
plt.show()

# Step-wise statistics for reference
step_stats = df.groupby("Step")[["X", "Y", "Z"]].agg(["mean", "std", "min", "max"])
print("\nStep-wise Statistics (rounded to 2 decimals):\n")
print(step_stats.round(2))

## QUESTION 3 ##
# Correlation matrix
print("\n---- QUESTION 3: CORRELATION MATRIX ----\n")
corr = df.corr(method="pearson")
print(corr, "\n")
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

## QUESTION 4 ##
# Define variables for ML models
X = df[["X", "Y", "Z"]]
y = df["Step"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model 1: Random Forest (GridSearchCV)
mdl1 = RandomForestClassifier(random_state=42)
mdl1_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}
mdl1_grid = GridSearchCV(mdl1, mdl1_params, cv=5, n_jobs=-1, scoring="accuracy")
mdl1_grid.fit(X_train, y_train)

print("\n---- QUESTION 4 ----\n")
print("---- MODEL 1 (Random Forest - GridSearchCV) ----\n")
print("Best Parameters:")
for param, value in mdl1_grid.best_params_.items():
    print(f" {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl1_grid.best_estimator_.predict(X_train)))
print("Testing Accuracy :", accuracy_score(y_test, mdl1_grid.best_estimator_.predict(X_test)), "\n")

# Model 2: Logistic Regression (GridSearchCV)
mdl2 = LogisticRegression(max_iter=5000)
mdl2_params = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"],
}
mdl2_grid = GridSearchCV(mdl2, mdl2_params, cv=5, n_jobs=-1, scoring="accuracy")
mdl2_grid.fit(X_train, y_train)

print("---- MODEL 2 (Logistic Regression - GridSearchCV) ----\n")
print("Best Parameters:")
for param, value in mdl2_grid.best_params_.items():
    print(f" {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl2_grid.best_estimator_.predict(X_train)))
print("Testing Accuracy :", accuracy_score(y_test, mdl2_grid.best_estimator_.predict(X_test)), "\n")

# Model 3: SVM (GridSearchCV)
mdl3 = SVC()
mdl3_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],
}
mdl3_grid = GridSearchCV(mdl3, mdl3_params, cv=5, n_jobs=-1, scoring="accuracy")
mdl3_grid.fit(X_train, y_train)

print("---- MODEL 3 (SVM - GridSearchCV) ----\n")
print("Best Parameters:")
for param, value in mdl3_grid.best_params_.items():
    print(f" {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl3_grid.best_estimator_.predict(X_train)))
print("Testing Accuracy :", accuracy_score(y_test, mdl3_grid.best_estimator_.predict(X_test)), "\n")

# Model 4: Random Forest (RandomizedSearchCV)
mdl4 = RandomForestClassifier(random_state=42)
mdl4_params = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", None],
}
mdl4_random = RandomizedSearchCV(mdl4, mdl4_params, n_iter=30, cv=5, n_jobs=-1, scoring="accuracy")
mdl4_random.fit(X_train, y_train)

print("---- MODEL 4 (Random Forest - RandomizedSearchCV) ----\n")
print("Best Parameters:")
for param, value in mdl4_random.best_params_.items():
    print(f" {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl4_random.best_estimator_.predict(X_train)))
print("Testing Accuracy :", accuracy_score(y_test, mdl4_random.best_estimator_.predict(X_test)), "\n")

## QUESTION 5 ##
# Model comparison
print("\n---- QUESTION 5: MODEL COMPARISON ----\n")
mdls = {
    "Random Forest (GS)": mdl1_grid.best_estimator_,
    "Logistic Regression": mdl2_grid.best_estimator_,
    "SVM": mdl3_grid.best_estimator_,
    "Random Forest (RS)": mdl4_random.best_estimator_,
}
results = {}
for name, mdl in mdls.items():
    y_pred = mdl.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
    }

for mdl, metrics in results.items():
    print(f"{mdl}:")
    for metric, value in metrics.items():
        print(f" {metric}: {value:.4f}")
    print()

# Best model
best_mdl = mdls[max(results, key=lambda x: results[x]["F1 Score"])]
print("Best Model Selected:", best_mdl, "\n")

y_pred_best = best_mdl.predict(X_test)
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## QUESTION 6 ##
# Stacking classifier
stacked_mdl = StackingClassifier(
    estimators=[
        ("Random Forest", mdl1_grid.best_estimator_),
        ("SVM", mdl3_grid.best_estimator_),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1,
)
stacked_mdl.fit(X_train, y_train)
y_pred_stacked = stacked_mdl.predict(X_test)
stacked_results = {
    "Accuracy": accuracy_score(y_test, y_pred_stacked),
    "Precision": precision_score(y_test, y_pred_stacked, average="weighted"),
    "Recall": recall_score(y_test, y_pred_stacked, average="weighted"),
    "F1 Score": f1_score(y_test, y_pred_stacked, average="weighted"),
}

print("\n---- QUESTION 6: STACKING CLASSIFIER ----\n")
for metric, value in stacked_results.items():
    print(f"{metric}: {value:.4f}")

cm_stack = confusion_matrix(y_test, y_pred_stacked)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Stacking Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## QUESTION 7 ##
# Save best model
joblib.dump(best_mdl, "best_model.joblib")
print("\n---- QUESTION 7: SAVING & TESTING MODEL ----\n")
print("Best model saved as best_model.joblib\n")

# Load model & predict new coordinates
loaded_model = joblib.load("best_model.joblib")
coords = pd.DataFrame(
    [
        [9.375, 3.0625, 1.51],
        [6.995, 5.125, 0.3875],
        [0, 3.0625, 1.93],
        [9.4, 3, 1.8],
        [9.4, 3, 1.3],
    ],
    columns=["X", "Y", "Z"],
)
predictions = loaded_model.predict(coords)
print("Predicted Steps for Given Coordinates:")
for c, p in zip(coords.values, predictions):
    print(f"Coordinates {c} --> Step {p}")
