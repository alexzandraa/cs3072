import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

# 1. Load Data and Prepare Features/Target
data_path = "assistant/data/synthetic_recruitment_data.csv"
df = pd.read_csv(data_path)

# Encode Gender as binary: Male=1, Female=0
df["Gender_binary"] = df["Gender"].map({"Male": 1, "Female": 0})

# Select features (including gender) and target
feature_names = ["Experience", "Test_Score", "Interview_Score", "Gender_binary"]
X = df[feature_names].values
y = df["Hired"].values

# 2. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test, train_info, test_info = train_test_split(
    X, y, df, test_size=0.3, random_state=42, stratify=y
)

# 3. Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train an MLP Classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Get predicted probabilities on training and test sets
y_train_probs = mlp_model.predict_proba(X_train_scaled)[:, 1]
y_test_probs = mlp_model.predict_proba(X_test_scaled)[:, 1]

# Optimise Threshold Based on F1 Score
threshold_candidates = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_f1 = 0.0
grid_results = []

for t in threshold_candidates:
    y_pred_t = (y_train_probs >= t).astype(int)
    r = recall_score(y_train, y_pred_t)
    p = precision_score(y_train, y_pred_t)
    f1 = f1_score(y_train, y_pred_t)
    grid_results.append((t, r, p, f1))
    
grid_df = pd.DataFrame(grid_results, columns=["Threshold", "Recall", "Precision", "F1 Score"])
print("=== Threshold Grid (Recall, Precision, F1 Score) ===")
print(grid_df.to_string(index=False))

best_idx = np.argmax(grid_df["F1 Score"].values)
best_threshold = grid_df.iloc[best_idx]["Threshold"]
print("\nBest threshold for maximizing F1:", best_threshold)

# Apply the optimized threshold on the test set
y_test_pred = (y_test_probs >= best_threshold).astype(int)

# Evaluate Overall Model Performance
cm_overall = confusion_matrix(y_test, y_test_pred)
print("\n=== Confusion Matrix (Overall) ===")
print(cm_overall)
print("\n=== Classification Report (Overall) ===")
print(classification_report(y_test, y_test_pred, digits=3))

acc = accuracy_score(y_test, y_test_pred)
f1_val = f1_score(y_test, y_test_pred)
rec_val = recall_score(y_test, y_test_pred)
prec_val = precision_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_probs)

print("Accuracy:", acc)
print("F1 Score:", f1_val)
print("Recall:", rec_val)
print("Precision:", prec_val)
print("ROC AUC:", auc)

#  Per-Gender Analysis
# Create a DataFrame from X_test_scaled for easier grouping
df_test = pd.DataFrame(X_test, columns=feature_names)
df_test["Hired"] = y_test
df_test["Prediction"] = y_test_pred
df_test["Predicted_Probability"] = y_test_probs

for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
    subset = df_test[df_test["Gender_binary"] == gender_val]
    print(f"\n--- {gender_label} ---")
    print("Confusion Matrix:\n", confusion_matrix(subset["Hired"], subset["Prediction"]))
    print("Classification Report:\n", classification_report(subset["Hired"], subset["Prediction"], digits=3))
    print("Mean Predicted Probability:", np.mean(subset["Predicted_Probability"]))


# Bias Analysis by Binning Candidate Features
df_test["Experience_bin"] = pd.cut(df_test["Experience"], bins=3, labels=["Low", "Medium", "High"])
df_test["Test_Score_bin"] = pd.cut(df_test["Test_Score"], bins=3, labels=["Low", "Medium", "High"])
df_test["Interview_Score_bin"] = pd.cut(df_test["Interview_Score"], bins=3, labels=["Low", "Medium", "High"])

for feature, bin_label in [("Experience", "Experience_bin"),
                           ("Test_Score", "Test_Score_bin"),
                           ("Interview_Score", "Interview_Score_bin")]:
    print(f"\n=== Bias Analysis by {feature} ===")
    group_stats = df_test.groupby([bin_label, "Gender_binary"])["Predicted_Probability"].mean().unstack()
    print(group_stats)
    group_stats.plot(kind="bar", title=f"Avg Predicted Probability by {feature} Bin and Gender")
    plt.xlabel(f"{feature} Bin")
    plt.ylabel("Avg Predicted Probability")
    plt.show()

# Evaluate Predictions on Identical Candidate Profiles
candidate_features = {
    'Experience': 5,
    'Test_Score': 50,
    'Interview_Score': 5
}
candidate_df = pd.DataFrame({
    'Experience': [candidate_features['Experience']] * 2,
    'Test_Score': [candidate_features['Test_Score']] * 2,
    'Interview_Score': [candidate_features['Interview_Score']] * 2,
    'Gender_binary': [1, 0]  # Male then Female
})
candidate_df_scaled = scaler.transform(candidate_df)
identical_probs = mlp_model.predict_proba(candidate_df_scaled)[:, 1]
print("\n=== Predicted Hiring Probabilities for Identical Candidates ===")
print("Male Candidate:", identical_probs[0])
print("Female Candidate:", identical_probs[1])
print("Difference:", abs(identical_probs[0] - identical_probs[1]))


# Save the MLP Model and Scaler
save_dir = "assistant/models"
os.makedirs(save_dir, exist_ok=True)
model_filepath = os.path.join(save_dir, "mlp_classifier.pkl")
scaler_filepath = os.path.join(save_dir, "mlp_scaler.pkl")

dump(mlp_model, model_filepath)
dump(scaler, scaler_filepath)

print("\nMLP model saved to:", model_filepath)
print("Scaler saved to:", scaler_filepath)
