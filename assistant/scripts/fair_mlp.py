import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score, confusion_matrix, classification_report)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric

import warnings
warnings.filterwarnings("ignore")

# Load and Prepare the Data
data_path = "assistant/data/synthetic_recruitment_data.csv"
df = pd.read_csv(data_path)

# Ensure the sensitive attribute is encoded as binary (Male=1, Female=0)
if 'Gender_binary' not in df.columns:
    df["Gender_binary"] = df["Gender"].map({"Male": 1, "Female": 0})


#  Split Data into Training and Test Sets
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Hired"])


#  Convert to AIF360 BinaryLabelDataset
train_dataset = BinaryLabelDataset(
    df=train_df[['Hired', 'Gender_binary']],
    label_names=['Hired'],
    protected_attribute_names=['Gender_binary']
)
test_dataset = BinaryLabelDataset(
    df=test_df[['Hired', 'Gender_binary']],
    label_names=['Hired'],
    protected_attribute_names=['Gender_binary']
)


# Apply Reweighing to the Training Data
privileged_groups = [{'Gender_binary': 1}]
unprivileged_groups = [{'Gender_binary': 0}]
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
train_dataset_transf = RW.fit_transform(train_dataset)


# Prepare Data for Model Training
feature_names = ["Experience", "Test_Score", "Interview_Score", "Gender_binary"]
X_train = train_df[feature_names].values
y_train = train_df["Hired"].values
sample_weights = train_dataset_transf.instance_weights

X_test = test_df[feature_names].values
y_test = test_df["Hired"].values


# Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train a Fair MLP Classifier via Resampling
# Resample the training data based on sample_weights.
# This effectively simulates using sample weights in a model that doesn't accept them.
sample_prob = sample_weights / sample_weights.sum()
resample_indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True, p=sample_prob)
X_train_resampled = X_train_scaled[resample_indices]
y_train_resampled = y_train[resample_indices]

mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp_model.fit(X_train_resampled, y_train_resampled)


# Predict Probabilities and Optimise Threshold
y_train_probs = mlp_model.predict_proba(X_train_scaled)[:, 1]
y_test_probs = mlp_model.predict_proba(X_test_scaled)[:, 1]

threshold_candidates = np.linspace(0.1, 0.9, 81)
grid_results = []
best_threshold = 0.5
best_f1 = 0.0

for t in threshold_candidates:
    y_pred_t = (y_train_probs >= t).astype(int)
    current_f1 = f1_score(y_train, y_pred_t)
    grid_results.append((t, recall_score(y_train, y_pred_t), precision_score(y_train, y_pred_t), current_f1))
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = t

print("=== Threshold Grid (Recall, Precision, F1 Score) ===")
print(pd.DataFrame(grid_results, columns=["Threshold", "Recall", "Precision", "F1 Score"]).to_string(index=False))
print("\nBest threshold for maximizing F1:", best_threshold)

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

# Evaluate Per-Gender Performance
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
identical_probs = mlp_model.predict_proba(candidate_df_scaled)[:, 1]
print("\n=== Predicted Hiring Probabilities for Identical Candidates ===")
print("Male Candidate:", identical_probs[0])
print("Female Candidate:", identical_probs[1])
print("Difference:", abs(identical_probs[0] - identical_probs[1]))


# 13. Save the Fair MLP Model and the Scaler
save_dir = "assistant/models"
os.makedirs(save_dir, exist_ok=True)

model_filepath = os.path.join(save_dir, "fair_mlp_model.pkl")
scaler_filepath = os.path.join(save_dir, "fair_mlp_scaler.pkl")

dump(mlp_model, model_filepath)
dump(scaler, scaler_filepath)

print("\nFair MLP model saved to:", model_filepath)
print("Scaler saved to:", scaler_filepath)
