import pandas as pd
import numpy as np
import joblib
import json
import time

from pipeline_utils import IQRCapper
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve
)

import warnings
warnings.filterwarnings("ignore")

t0 = time.time()

# 1. LOAD
DATA_PATH = "predictive_maintenance_dataset.csv"  # place the dataset in this folder, or update the path
df = pd.read_csv(DATA_PATH)
TARGET = "Machine failure"

identifier_columns = ["UDI", "Product ID"]
potential_leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
columns_to_drop = [c for c in identifier_columns + potential_leakage_columns if c in df.columns]
df = df.drop(columns=columns_to_drop, errors="ignore")

# 2. FEATURE ENGINEERING
df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
df["Strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_columns),
    ("numeric", "passthrough", numeric_columns)
], remainder="drop")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF": Pipeline([("iqr", IQRCapper()), ("preprocessor", preprocessor),
                    ("model", RandomForestClassifier(random_state=42, n_jobs=-1))]),
    "DT": Pipeline([("iqr", IQRCapper()), ("preprocessor", preprocessor),
                    ("model", DecisionTreeClassifier(random_state=42))]),
    "KNN": Pipeline([("iqr", IQRCapper()), ("preprocessor", preprocessor),
                     ("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
    "LR": Pipeline([("iqr", IQRCapper()), ("preprocessor", preprocessor),
                    ("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42, max_iter=2000))])
}

params = {
    "RF": {"model__n_estimators": [100, 200], "model__max_depth": [10, 15],
           "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2],
           "model__class_weight": ["balanced"]},
    "DT": {"model__criterion": ["gini", "entropy"], "model__max_depth": [5, 10, 15],
           "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2],
           "model__class_weight": ["balanced"]},
    "KNN": {"model__n_neighbors": [5, 7, 11], "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]},
    "LR": {"model__C": [0.1, 1, 10], "model__penalty": ["l1", "l2"],
           "model__solver": ["liblinear"], "model__class_weight": ["balanced"]}
}

best_models = {}
model_results = []

for name in models:
    grid = GridSearchCV(models[name], params[name], cv=cv_strategy, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    model_results.append({"Model": name, "CV F1": round(grid.best_score_, 4), "Best Params": grid.best_params_})
    print(f"{name}: CV F1 = {grid.best_score_:.4f} | {grid.best_params_}")

stack_estimators = [(k.lower(), clone(v)) for k, v in best_models.items()]
meta_model = LogisticRegression(C=10, penalty="l2", solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=42)
stack = StackingClassifier(estimators=stack_estimators, final_estimator=meta_model,
                            cv=cv_strategy, passthrough=False, stack_method="predict_proba", n_jobs=-1)
stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
y_proba = stack.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": average_precision_score(y_test, y_proba),
}
cm = confusion_matrix(y_test, y_pred).tolist()
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

print("\nFINAL METRICS:", metrics)
print("Confusion matrix:", cm)

# SAVE ARTIFACTS — copy these three files into ../app/ to update the deployed dashboard
joblib.dump(stack, "stack_model.joblib")
joblib.dump(best_models, "base_models.joblib")

roc_data = {"fpr": fpr.tolist()[::max(1, len(fpr)//200)], "tpr": tpr.tolist()[::max(1, len(tpr)//200)]}

report = {
    "metrics": metrics,
    "confusion_matrix": cm,
    "model_results": model_results,
    "roc_curve": roc_data,
    "categorical_columns": categorical_columns,
    "numeric_columns": numeric_columns,
    "feature_columns": X.columns.tolist(),
    "class_balance": y.value_counts().to_dict(),
    "n_samples": len(df),
}
with open("model_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nDone in {time.time()-t0:.1f}s")
