import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import json
import os

# ---- SETTINGS ----
DATASET_PATH = "Spam Detection.csv"  # Put your dataset in the main folder
FIG_DIR = "output/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---- LOAD DATA ----
df = pd.read_csv(DATASET_PATH)

# ---- EDA: Class distribution ----
plt.figure(figsize=(6,4))
sns.countplot(x=df['spam'])
plt.title("Class Distribution")
plt.savefig(f"{FIG_DIR}/fig_class_distribution.png")
plt.close()

# ---- FEATURES & LABEL ----
X = df.drop(columns=['spam'])
y = df['spam']

# ---- SCALING ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- TRAIN-TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---- PARAMETER GRIDS ----
param_grids = {
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3,5,7], 'weights': ['uniform', 'distance']}
    },
    'log_reg': {
        'model': LogisticRegression(max_iter=500),
        'params': {'C': [0.1, 1, 10]}
    },
    'rf': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [100,200], 'max_depth': [None, 10, 20]}
    }
}

results = {}

# ---- TRAIN & EVALUATE ----
for name, mp in param_grids.items():
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'best_params': grid.best_params_,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{FIG_DIR}/cm_{name}.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"{name.upper()} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name.upper()}")
    plt.legend()
    plt.savefig(f"{FIG_DIR}/roc_{name}.png")
    plt.close()

# ---- METRICS BAR CHART ----
metrics_df = pd.DataFrame(results).T[['accuracy','precision','recall','f1_score']]
metrics_df.plot(kind='bar', figsize=(8,5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.savefig(f"{FIG_DIR}/metrics_bar.png")
plt.close()

# ---- SAVE RESULTS ----
with open("output/experiment_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("DONE! Figures are saved in output/figures/ and metrics in output/experiment_results.json")
