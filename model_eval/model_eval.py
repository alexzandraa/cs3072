import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

# configuration
DATA_PATH = "assistant/data/synthetic_recruitment_data.csv"
MODEL_DIR = "assistant/models"
PLOT_DIR = "assistant/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURES = ["Experience", "Test_Score", "Interview_Score", "Gender_binary"]
TARGET = "Hired"

MODELS = {
    "Biased Logistic Regression": {
        "model": os.path.join(MODEL_DIR, "biased_logistic_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, "logistic_scaler.pkl"),
        "fair": False
    },
    "Fair Logistic Regression": {
        "model": os.path.join(MODEL_DIR, "fair_logistic_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, "fair_logistic_scaler.pkl"),
        "fair": True
    },
    "Biased Random Forest": {
        "model": os.path.join(MODEL_DIR, "biased_random_forest_model.pkl"),
        "scaler": None,
        "fair": False
    },
    "Biased XGBoost": {
        "model": os.path.join(MODEL_DIR, "biased_xgb_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, "xgb_scaler.pkl"),
        "fair": False
    },
    "Biased MLP": {
        "model": os.path.join(MODEL_DIR, "mlp_classifier.pkl"),
        "scaler": os.path.join(MODEL_DIR, "mlp_scaler.pkl"),
        "fair": False
    },
    "Fair MLP": {
        "model": os.path.join(MODEL_DIR, "fair_mlp_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, "mlp_scaler.pkl"),
        "fair": True
    }
}

# load Data
df = pd.read_csv(DATA_PATH)
df["Gender_binary"] = df["Gender"].map({"Male": 1, "Female": 0})

X = df[FEATURES]
y = df[TARGET]

# evaluate models
for name, config in MODELS.items():
    print(f"\n{name} Evaluation Report")
    print("-" * 50)

    model = joblib.load(config["model"])
    scaler = joblib.load(config["scaler"]) if config["scaler"] else None

    X_eval = scaler.transform(X) if scaler else X.values
    y_probs = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_probs)

    print(classification_report(y, y_pred, digits=4))
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # per-gender 
    print("Gender Breakdown ")
    for g_val, g_name in [(0, "Female"), (1, "Male")]:
        mask = df["Gender_binary"] == g_val
        y_true = y[mask]
        y_pred_g = y_pred[mask]
        print(f"\nGender: {g_name}")
        print(confusion_matrix(y_true, y_pred_g))
        print(classification_report(y_true, y_pred_g, digits=4))

    # probability distribution plot 
    if "Logistic" in name:
        df[f"prob_{name}"] = y_probs
        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=df, x=f"prob_{name}", hue="Gender", common_norm=False)
        plt.title(f"Prediction Distribution by Gender — {name}")
        plt.xlabel("Predicted Hiring Probability")
        plt.ylabel("Density")
        plot_path = os.path.join(PLOT_DIR, f"{name.lower().replace(' ', '_')}_prob_dist.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved gender probability plot to: {plot_path}")
