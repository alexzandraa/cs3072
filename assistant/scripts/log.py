import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


data_path = "assistant/data/synthetic_candidate_pool.csv"
df = pd.read_csv(data_path)


features = ["Occupation", "Age_Bracket", "Gender", "Ethnicity", "Age", "Skill_Score"]
target = "Hired"


df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


X_train = df_train[features].copy()
y_train = df_train[target].copy()
X_test = df_test[features].copy()
y_test = df_test[target].copy()


categorical_features = ["Occupation", "Age_Bracket", "Gender", "Ethnicity"]


X_train_enc = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)


X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns, index=X_test_enc.index)


log_reg = LogisticRegression(random_state=42, max_iter=500)
log_reg.fit(X_train_scaled, y_train)


y_test_pred = log_reg.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test Recall:", recall_score(y_test, y_test_pred))
print("Test F1 Score:", f1_score(y_test, y_test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))


conf_mat = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


model_folder = "assistant/models"
os.makedirs(model_folder, exist_ok=True)
joblib.dump(log_reg, os.path.join(model_folder, "logistic_model.pkl"))
joblib.dump(scaler, os.path.join(model_folder, "scaler_logistic.pkl"))
joblib.dump(X_train_enc.columns.tolist(), os.path.join(model_folder, "train_columns_logistic.pkl"))
print("Logistic model, scaler, and training columns saved successfully under 'assistant/models'.")


def predict_candidate_logistic(new_candidate_dict):
    """
    Given a dictionary for a new candidate with keys matching the features:
      - Occupation, Age_Bracket, Gender, Ethnicity, Age, Skill_Score
    This function preprocesses the input using the saved scaler and training columns,
    then uses the saved logistic model to predict whether the candidate would be hired.
    Returns a dictionary with:
      - 'Hired': Boolean prediction
      - 'Probability': Predicted hiring probability.
    """
    df_new = pd.DataFrame([new_candidate_dict])
    df_new_enc = pd.get_dummies(df_new, columns=categorical_features, drop_first=True)
    train_columns = joblib.load(os.path.join(model_folder, "train_columns_logistic.pkl"))
    df_new_enc = df_new_enc.reindex(columns=train_columns, fill_value=0)
    scaler_loaded = joblib.load(os.path.join(model_folder, "scaler_logistic.pkl"))
    df_new_scaled = pd.DataFrame(scaler_loaded.transform(df_new_enc), columns=train_columns)
    model_loaded = joblib.load(os.path.join(model_folder, "logistic_model.pkl"))
    prob = model_loaded.predict_proba(df_new_scaled)[0][1]
    pred = model_loaded.predict(df_new_scaled)[0]
    return {"Hired": bool(pred), "Probability": prob}


new_candidate = {
    "Occupation": "2131 IT project managers",
    "Age_Bracket": "30 to 34",
    "Gender": "Male",
    "Ethnicity": "White",
    "Age": 32,
    "Skill_Score": 65
}

prediction = predict_candidate_logistic(new_candidate)
print("\nNew Candidate Prediction (Logistic Regression):")
print(prediction)
