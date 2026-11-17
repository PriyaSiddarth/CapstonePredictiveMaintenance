from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix
)

project_dir = Path("CapstonePredictiveMaintenance")

model_path = (
    project_dir / "deployment" /
    "best_engine_condition_pipeline.joblib"
)

test_dir = project_dir / "data" / "processed"

# Load the exact model trained by the pipeline
model = joblib.load(model_path)

# Use held-out test data, not the training data
Xtest = pd.read_csv(test_dir / "Xtest.csv")
ytest = pd.read_csv(test_dir / "ytest.csv").squeeze("columns")

# Ensure column names match model training
print("Test dataset columns:", Xtest.columns.tolist())

# Original precision
X_original = Xtest.copy()

# Rounded to 2 decimal places
X_rounded = Xtest.round(2)

# Use the same threshold as model evaluation
threshold = 0.45

original_pred = (
    model.predict_proba(X_original)[:, 1] >= threshold
).astype(int)

rounded_pred = (
    model.predict_proba(X_rounded)[:, 1] >= threshold
).astype(int)

# Compare results
original_accuracy = accuracy_score(ytest, original_pred)
rounded_accuracy = accuracy_score(ytest, rounded_pred)

original_recall = recall_score(ytest, original_pred)
rounded_recall = recall_score(ytest, rounded_pred)

changed = (original_pred != rounded_pred).sum()

agreement = (
    original_pred == rounded_pred
).mean() * 100

print("\n========== ROUNDING VALIDATION ==========")

print(f"Original Accuracy : {original_accuracy:.4f}")
print(f"Rounded Accuracy  : {rounded_accuracy:.4f}")

print(f"Original Recall   : {original_recall:.4f}")
print(f"Rounded Recall    : {rounded_recall:.4f}")

print(f"Predictions Changed : {changed}")
print(f"Prediction Agreement: {agreement:.2f}%")

print("\nOriginal Confusion Matrix:")
print(confusion_matrix(ytest, original_pred))

print("\nRounded Confusion Matrix:")
print(confusion_matrix(ytest, rounded_pred))

print("========================================")
