import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score, confusion_matrix)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt
import seaborn as sns


# Load and Prepare the Data
data_path = "assistant/data/synthetic_recruitment_data.csv"
df = pd.read_csv(data_path)

# Ensure the sensitive attribute is encoded as binary (Male=1, Female=0)
if 'Gender_binary' not in df.columns:
    df["Gender_binary"] = df["Gender"].map({"Male": 1, "Female": 0})

# 2. Split Data into Training and Test Sets
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


#  Prepare Data for Model Training
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

# Train a Logistic Regression Model Using Sample Weights
base_clf = LogisticRegression(solver='liblinear', random_state=42)
base_clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# 8. Calibrate the Classifier (Platt Scaling)
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Get calibrated predicted probabilities on the test set
y_test_probs = calibrated_clf.predict_proba(X_test_scaled)[:, 1]

# Optimise Threshold Based on F1 Score
threshold_candidates = np.linspace(0.1, 0.7, 61)
grid_results = []
for t in threshold_candidates:
    y_pred_t = (y_test_probs >= t).astype(int)
    r = recall_score(y_test, y_pred_t)
    p = precision_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    grid_results.append((t, r, p, f1))
grid_df = pd.DataFrame(grid_results, columns=["Threshold", "Recall", "Precision", "F1 Score"])
print("=== Threshold Grid (Recall, Precision, F1 Score) ===")
print(grid_df.to_string(index=False))

# Choose threshold that maximizes F1 score
best_idx = np.argmax(grid_df["F1 Score"].values)
best_threshold = grid_df.iloc[best_idx]["Threshold"]
print("\nBest threshold for maximizing F1:", best_threshold)

# Apply optimised threshold
y_test_pred = (y_test_probs >= best_threshold).astype(int)

# Evaluate Overall Model Performance
acc = accuracy_score(y_test, y_test_pred)
f1_val = f1_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_probs)

print("\n=== Overall Model Performance (Optimized Threshold) ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1_val)
print("ROC AUC:", auc)

# 11. Evaluate Confusion Matrices by Gender
print("\n=== Confusion Matrices by Gender ===")
for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
    mask = test_df["Gender_binary"] == gender_val
    cm_gender = confusion_matrix(y_test[mask], y_test_pred[mask])
    print(f"\nConfusion Matrix for {gender_label}:")
    print(cm_gender)
    acc_gender = np.mean(y_test[mask] == y_test_pred[mask])
    print(f"Accuracy for {gender_label}:", acc_gender)

# Evaluate Fairness Metrics on Test Set
test_dataset_pred = test_dataset.copy(deepcopy=True)
test_dataset_pred.labels = y_test_pred.reshape(-1, 1)

metric_test = ClassificationMetric(
    test_dataset,
    test_dataset_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n=== Fairness Metrics on Test Set ===")
print("Statistical Parity Difference: {:.3f}".format(metric_test.statistical_parity_difference()))
print("Equal Opportunity Difference: {:.3f}".format(metric_test.equal_opportunity_difference()))
print("Average Odds Difference: {:.3f}".format(metric_test.average_odds_difference()))

# 12. Evaluate Predictions on Identical Candidate Profiles
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
candidate_df_scaled = scaler.transform(candidate_df[feature_names])
candidate_probs = calibrated_clf.predict_proba(candidate_df_scaled)[:, 1]
print("\n=== Predicted Hiring Probabilities for Identical Candidates ===")
print("Male Candidate:", candidate_probs[0])
print("Female Candidate:", candidate_probs[1])
print("Difference:", abs(candidate_probs[0] - candidate_probs[1]))


# Visualise the Distribution of Predicted Probabilities by Gender
test_df = test_df.copy()
test_df['Predicted_Probability'] = y_test_probs

plt.figure(figsize=(8, 6))
sns.kdeplot(data=test_df, x='Predicted_Probability', hue='Gender_binary', common_norm=False)
plt.title('Distribution of Calibrated Predicted Probabilities by Gender')
plt.xlabel('Predicted Probability of Hiring')
plt.ylabel('Density')
plt.show()

# Save the Calibrated Model and the Scaler
save_folder = "assistant/models"
os.makedirs(save_folder, exist_ok=True)

model_filepath = os.path.join(save_folder, "fair_logistic_model.pkl")
scaler_filepath = os.path.join(save_folder, "fair_logistic_scaler.pkl")

joblib.dump(calibrated_clf, model_filepath)
joblib.dump(scaler, scaler_filepath)

print(f"Model saved to {model_filepath}")
print(f"Scaler saved to {scaler_filepath}")
