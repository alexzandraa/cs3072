import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import shap

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


data_path = "assistant/data/synthetic_candidate_pool.csv"
df = pd.read_csv(data_path)


df['Gender_binary'] = df['Gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)


df_numeric = df[['Age', 'Skill_Score', 'Hired', 'Gender_binary']].copy()


dataset = BinaryLabelDataset(
    df=df_numeric,
    label_names=['Hired'],
    protected_attribute_names=['Gender_binary'],
    favorable_label=1,
    unfavorable_label=0
)


privileged_groups = [{'Gender_binary': 1}]
unprivileged_groups = [{'Gender_binary': 0}]
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset)
sample_weights = dataset_transf.instance_weights


X = df_numeric[['Age', 'Skill_Score']].values
y = df_numeric['Hired'].values


rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, class_weight='balanced')
rf.fit(X, y, sample_weight=sample_weights)


y_pred = rf.predict(X)
print("Model Evaluation on Full Data:")
print(classification_report(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)


print("X shape:", X.shape)
print("SHAP values for class 1 shape:", shap_values[1].shape)


shap.summary_plot(shap_values[1], X, feature_names=['Age', 'Skill_Score'])


model_folder = "assistant/models"
os.makedirs(model_folder, exist_ok=True)
joblib.dump(rf, os.path.join(model_folder, "random_forest_model_fair.pkl"))

joblib.dump(['Age', 'Skill_Score'], os.path.join(model_folder, "features_rf_fair.pkl"))
print("Fair Random Forest model and artifacts saved successfully under 'assistant/models'.")


def predict_candidate_fair(new_candidate_dict):
    """
    Given a dictionary with keys 'Age' and 'Skill_Score',
    this function preprocesses the input and returns a hiring prediction and probability
    from the fair Random Forest model.
    """
    df_new = pd.DataFrame([new_candidate_dict])
    features = joblib.load(os.path.join(model_folder, "features_rf_fair.pkl"))
    X_new = df_new[features].values
    model_loaded = joblib.load(os.path.join(model_folder, "random_forest_model_fair.pkl"))
    prob = model_loaded.predict_proba(X_new)[0][1]
    pred = model_loaded.predict(X_new)[0]
    return {"Hired": bool(pred), "Probability": prob}


new_candidate = {
    "Age": 32,
    "Skill_Score": 65
}

prediction = predict_candidate_fair(new_candidate)
print("\nNew Candidate Prediction (Fair Random Forest):")
print(prediction)
