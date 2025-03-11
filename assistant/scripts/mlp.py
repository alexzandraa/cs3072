import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.stats import uniform

warnings.filterwarnings("ignore", category=UserWarning)


data_path = "assistant/data/synthetic_candidate_pool.csv"
df = pd.read_csv(data_path)


base_features = [
    "Occupation", "Age_Bracket", "Gender", "Ethnicity",
    "Education_Level", "Age", "Experience", "Skill_Score", "Quality"
]
features = base_features

target = "Hired"


df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df[target]
)

df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=42, stratify=df_temp[target]
)

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)


X_train = df_train[features].copy()
y_train = df_train[target].copy()
X_val   = df_val[features].copy()
y_val   = df_val[target].copy()
X_test  = df_test[features].copy()
y_test  = df_test[target].copy()


categorical_features = [
    "Occupation", "Age_Bracket", "Gender", "Ethnicity", "Education_Level"
]
numeric_features = list(set(features) - set(categorical_features))


X_train_enc = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_val_enc   = pd.get_dummies(X_val, columns=categorical_features, drop_first=True)
X_test_enc  = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)


X_val_enc  = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val_enc), columns=X_val_enc.columns, index=X_val_enc.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_enc), columns=X_test_enc.columns, index=X_test_enc.index
)


param_distributions = {
    "hidden_layer_sizes": [(50,), (100,), (150,), (100,50), (100,100)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "sgd"],
    "alpha": uniform(1e-4, 1e-2),   
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": uniform(1e-4, 1e-3)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mlp_base = MLPClassifier(
    max_iter=2000,
    tol=1e-4,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=mlp_base,
    param_distributions=param_distributions,
    n_iter=30,           
    cv=cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_scaled, y_train)
print("Best parameters:", random_search.best_params_)
print("Best cross-validation F1 score:", random_search.best_score_)

best_model = random_search.best_estimator_


y_val_pred = best_model.predict(X_val_scaled)
print("\nValidation Results:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("F1 Score:", f1_score(y_val, y_val_pred))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))


y_test_pred = best_model.predict(X_test_scaled)
print("\nTest Results:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 Score:", f1_score(y_test, y_test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))


conf_mat_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat_test, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


model_folder = "assistant/models"
os.makedirs(model_folder, exist_ok=True)

joblib.dump(best_model, os.path.join(model_folder, "mlp_model.pkl"))
joblib.dump(scaler, os.path.join(model_folder, "scaler_mlp.pkl"))
joblib.dump(X_train_enc.columns.tolist(), os.path.join(model_folder, "train_columns_mlp.pkl"))
print("\nMLP model, scaler, and training columns saved successfully under 'assistant/models'.")


def predict_candidate_mlp(new_candidate_dict):
    """
    Predict whether a new candidate would be hired.
    Expected keys: Occupation, Age_Bracket, Gender, Ethnicity, Education_Level,
    Age, Experience, Skill_Score, Quality.
    """
    
    train_columns = joblib.load(os.path.join(model_folder, "train_columns_mlp.pkl"))
    scaler_loaded = joblib.load(os.path.join(model_folder, "scaler_mlp.pkl"))
    model_loaded = joblib.load(os.path.join(model_folder, "mlp_model.pkl"))

    
    df_new = pd.DataFrame([new_candidate_dict])
    
    
    categorical_feats = ["Occupation", "Age_Bracket", "Gender", "Ethnicity", "Education_Level"]
    df_new_enc = pd.get_dummies(df_new, columns=categorical_feats, drop_first=True)
    df_new_enc = df_new_enc.reindex(columns=train_columns, fill_value=0)
    
    
    df_new_scaled = pd.DataFrame(
        scaler_loaded.transform(df_new_enc),
        columns=df_new_enc.columns
    )

    
    prob = model_loaded.predict_proba(df_new_scaled)[0][1]
    pred = model_loaded.predict(df_new_scaled)[0]
    return {"Hired": bool(pred), "Probability": prob}


new_candidate = {
    "Occupation": "2131 IT project managers",
    "Age_Bracket": "30 to 34",
    "Gender": "Male",
    "Ethnicity": "White",
    "Education_Level": "Undergraduate",
    "Age": 32,
    "Experience": 8,
    "Skill_Score": 60,
    "Quality": 7.0
}
prediction = predict_candidate_mlp(new_candidate)
print("\nNew Candidate Prediction (MLPClassifier):")
print(prediction)
