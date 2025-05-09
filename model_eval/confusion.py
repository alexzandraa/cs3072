import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# file paths
DATA_PATH = "assistant/data/synthetic_recruitment_data.csv"
PLOTS_DIR = "assistant/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# load and prepare Data
data = pd.read_csv(DATA_PATH)
data["Gender_binary"] = data["Gender"].map({"Male": 1, "Female": 0})

X = data[["Experience", "Test_Score", "Interview_Score", "Gender_binary"]]
y = data["Hired"]

X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
    X, y, data, test_size=0.3, random_state=42, stratify=y
)

# confusion matrix plot function 
def plot_conf_matrix(y_true, y_pred, gender_label, model_label, file_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hired", "Hired"])
    disp.plot(cmap="Blues", colorbar=True)
    plt.title(f"{model_label} — {gender_label}")
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# model evaluation runner 
def evaluate_model(model_path, scaler_path, label):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_test)
    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
        mask = (X_test["Gender_binary"] == gender_val)
        plot_conf_matrix(
            y_test[mask],
            y_pred[mask],
            gender_label,
            label,
            f"{label.lower().replace(' ', '_')}_{gender_label.lower()}_conf_new.png"
        )

# run evaluation 
evaluate_model(
    model_path="assistant/models/biased_logistic_model.pkl",
    scaler_path="assistant/models/logistic_scaler.pkl",
    label="Biased Logistic Regression"
)

evaluate_model(
    model_path="assistant/models/fair_logistic_model.pkl",
    scaler_path="assistant/models/fair_logistic_scaler.pkl",
    label="Fair Logistic Regression"
)
