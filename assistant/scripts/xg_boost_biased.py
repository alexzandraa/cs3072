import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# Load Data and Prepare Features/Target
data_path = "assistant/data/synthetic_recruitment_data.csv"  # Adjust if needed
data = pd.read_csv(data_path)

# Encode Gender as binary: Male=1, Female=0
data["Gender_binary"] = data["Gender"].map({"Male": 1, "Female": 0})

# Select features (including gender) and target
feature_names = ["Experience", "Test_Score", "Interview_Score", "Gender_binary"]
X = data[feature_names].values
y = data["Hired"].values


# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test, train_info, test_info = train_test_split(
    X, y, data, test_size=0.3, random_state=42, stratify=y
)


# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Biased XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predict probabilities on training and test sets
y_train_probs = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_test_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Optimise Classification Threshold Based on F1 Score
threshold_candidates = np.linspace(0.1, 0.9, 81)
grid_results = []
for t in threshold_candidates:
    y_pred_t = (y_train_probs >= t).astype(int)
    current_f1 = f1_score(y_train, y_pred_t)
    r = recall_score(y_train, y_pred_t)
    p = precision_score(y_train, y_pred_t)
    grid_results.append((t, r, p, current_f1))
grid_df = pd.DataFrame(grid_results, columns=["Threshold", "Recall", "Precision", "F1 Score"])
print("=== Threshold Grid (Recall, Precision, F1 Score) ===")
print(grid_df.to_string(index=False))

best_idx = np.argmax(grid_df["F1 Score"].values)
best_threshold = grid_df.iloc[best_idx]["Threshold"]
print("\nBest threshold for maximizing F1:", best_threshold)

# Apply optimised threshold on test set
y_test_pred = (y_test_probs >= best_threshold).astype(int)


# Evaluate Overall Model Performance
cm = confusion_matrix(y_test, y_test_pred)
print("\n=== Confusion Matrix (Overall) ===")
print(cm)

print("\n=== Classification Report (Overall) ===")
print(classification_report(y_test, y_test_pred, digits=3))

acc = accuracy_score(y_test, y_test_pred)
f1_val = f1_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_probs)

print("Accuracy:", acc)
print("F1 Score:", f1_val)
print("Recall:", rec)
print("Precision:", prec)
print("ROC AUC:", auc)

# Per-Gender Analysis (Confusion Matrix, Classification Report)
print("\n=== Confusion Matrices by Gender ===")
for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
    mask = (X_test[:, 3] == gender_val)  # Column 3 corresponds to Gender_binary
    y_true_gender = y_test[mask]
    y_pred_gender = y_test_pred[mask]
    
    print(f"\n--- Gender: {gender_label} ---")
    cm_gender = confusion_matrix(y_true_gender, y_pred_gender)
    print("Confusion Matrix:")
    print(cm_gender)
    print("Classification Report:")
    print(classification_report(y_true_gender, y_pred_gender, digits=3))
    acc_gender = np.mean(y_true_gender == y_pred_gender)
    print(f"Accuracy for {gender_label}:", acc_gender)

#  Plot Probability Distributions by Gender
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data, x='Hiring_Probability', hue='Gender', common_norm=False)
plt.title('Distribution of Hiring Probabilities by Gender')
plt.xlabel('Hiring Probability')
plt.ylabel('Density')
plt.show()


# Demonstrate Bias on Identical Candidate Profiles
candidate_features = {
    'Experience': 10,
    'Test_Score': 100,
    'Interview_Score': 10
}
candidate_df = pd.DataFrame({
    'Experience': [candidate_features['Experience']] * 2,
    'Test_Score': [candidate_features['Test_Score']] * 2,
    'Interview_Score': [candidate_features['Interview_Score']] * 2,
    'Gender_binary': [1, 0]  # Male then Female
})
candidate_df_scaled = scaler.transform(candidate_df)
candidate_probs = xgb_model.predict_proba(candidate_df_scaled)[:, 1]
print("\n=== Predicted Hiring Probabilities for Identical Candidates ===")
print("Male Candidate:", candidate_probs[0])
print("Female Candidate:", candidate_probs[1])
print("Difference:", abs(candidate_probs[0] - candidate_probs[1]))


# Save the Biased XGBoost Model and Scaler
save_dir = "assistant/models"
os.makedirs(save_dir, exist_ok=True)

model_filepath = os.path.join(save_dir, "biased_xgb_model.pkl")
scaler_filepath = os.path.join(save_dir, "xgb_scaler.pkl")

joblib.dump(xgb_model, model_filepath)
joblib.dump(scaler, scaler_filepath)

print(f"\nBiased XGBoost model saved to: {model_filepath}")
print(f"Scaler saved to: {scaler_filepath}")

