import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
from joblib import dump  # For saving the model

import warnings
warnings.filterwarnings("ignore")

# Load Data and Prepare Features/Target=
data_path = "assistant/data/synthetic_recruitment_data.csv"  
data = pd.read_csv(data_path)

# Encode Gender as binary: Male=1, Female=0
data["Gender_binary"] = data["Gender"].map({"Male": 1, "Female": 0})

# Select features (including Gender) and target
X = data[["Experience", "Test_Score", "Interview_Score", "Gender_binary"]]
y = data["Hired"]

# Train/Test Split (stratify by target)
X_train, X_test, y_train, y_test, train_info, test_info = train_test_split(
    X, y, data, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest with Hyperparameter Tuning
param_grid = {
    'max_depth': [4, 6, 8],           # limit depth of trees
    'min_samples_split': [10, 20],    # minimum samples required to split
    'n_estimators': [100]             # keep the number of trees fixed for now
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

rf_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Predict probabilities on training and test sets
y_train_probs = rf_model.predict_proba(X_train)[:, 1]
y_test_probs = rf_model.predict_proba(X_test)[:, 1]


# Optimise Classification Threshold
thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_f1 = 0.0

for thresh in thresholds:
    y_pred_train = (y_train_probs >= thresh).astype(int)
    f1_val = f1_score(y_train, y_pred_train)
    if f1_val > best_f1:
        best_f1 = f1_val
        best_threshold = thresh

print(f"Optimal Threshold on Training Set: {best_threshold:.2f}")
print(f"Best Training F1 Score: {best_f1:.3f}")

# Apply the best threshold to test set predictions
y_test_pred = (y_test_probs >= best_threshold).astype(int)

# Overall Model Analysis
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

# 5. Per-Gender Analysis (Confusion Matrix, Classification Report)
test_info = test_info.copy()
test_info["Predicted_Hired"] = y_test_pred

print("\n=== Confusion Matrices and Classification Reports by Gender ===")
for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
    mask = (X_test["Gender_binary"] == gender_val)
    y_true_gender = y_test[mask]
    y_pred_gender = y_test_pred[mask]
    
    print(f"\n--- Gender: {gender_label} ---")
    cm_gender = confusion_matrix(y_true_gender, y_pred_gender)
    print("Confusion Matrix:")
    print(cm_gender)
    
    print("\nClassification Report:")
    print(classification_report(y_true_gender, y_pred_gender, digits=3))


# Plot Hiring Probability Distributions by Gender
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data, x='Hiring_Probability', hue='Gender', common_norm=False)
plt.title('Distribution of Hiring Probabilities by Gender')
plt.xlabel('Hiring Probability')
plt.ylabel('Density')
plt.show()


#  Demonstrate Bias on Identical Candidate Profiles
candidate_features = {
    'Experience': 5,      # years of experience
    'Test_Score': 50,     # test score
    'Interview_Score': 5  # interview score
}

candidate_df = pd.DataFrame({
    'Experience': [candidate_features['Experience']] * 2,
    'Test_Score': [candidate_features['Test_Score']] * 2,
    'Interview_Score': [candidate_features['Interview_Score']] * 2,
    'Gender_binary': [1, 0]  # Male then Female
})

predicted_probs = rf_model.predict_proba(candidate_df)[:, 1]

print("\n=== Predicted Hiring Probabilities for Identical Candidates ===")
print("Male Candidate:", predicted_probs[0])
print("Female Candidate:", predicted_probs[1])

# Save the Random Forest Model
save_dir = "assistant/models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, "biased_random_forest_model.pkl")

# Save the random forest model using joblib
dump(rf_model, model_path)

print(f"\nBiased Random Forest model saved to: {model_path}")
