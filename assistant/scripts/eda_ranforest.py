import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data_path = "assistant/data/synthetic_candidate_pool.csv"
df = pd.read_csv(data_path)

features = ["Occupation", "Age_Bracket", "Gender", "Ethnicity", "Skill_Score"]
target = "Hired"


_, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
print("Test set shape:", df_test.shape)


model_folder = "assistant/models"
rf_model = joblib.load(os.path.join(model_folder, "random_forest_model.pkl"))
scaler_rf = joblib.load(os.path.join(model_folder, "scaler_rf.pkl"))
train_columns = joblib.load(os.path.join(model_folder, "train_columns_rf.pkl"))


categorical_features = ["Occupation", "Age_Bracket", "Gender", "Ethnicity"]
X_test = df_test[features].copy()
y_test = df_test[target].copy()


X_test_enc = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

X_test_enc = X_test_enc.reindex(columns=train_columns, fill_value=0)


X_test_scaled = pd.DataFrame(scaler_rf.transform(X_test_enc), 
                             columns=X_test_enc.columns, 
                             index=X_test_enc.index)


y_test_pred = rf_model.predict(X_test_scaled)
y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]


df_test['y_true'] = y_test.values
df_test['y_pred'] = y_test_pred
df_test['y_proba'] = y_test_proba


age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
df_test['Age_Group'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels)


def compute_group_metrics(group):
    return pd.Series({
        'Count': len(group),
        'Accuracy': accuracy_score(group['y_true'], group['y_pred']),
        'Precision': precision_score(group['y_true'], group['y_pred'], zero_division=0),
        'Recall': recall_score(group['y_true'], group['y_pred'], zero_division=0),
        'F1': f1_score(group['y_true'], group['y_pred'], zero_division=0)
    })


metrics_gender = df_test.groupby('Gender').apply(compute_group_metrics).reset_index()
metrics_ethnicity = df_test.groupby('Ethnicity').apply(compute_group_metrics).reset_index()
metrics_age = df_test.groupby('Age_Group').apply(compute_group_metrics).reset_index()

print("Metrics by Gender:\n", metrics_gender)
print("\nMetrics by Ethnicity:\n", metrics_ethnicity)
print("\nMetrics by Age Group:\n", metrics_age)


hiring_rate_gender = df_test.groupby('Gender')['y_pred'].mean().reset_index().rename(columns={'y_pred':'Hiring_Rate'})
hiring_rate_ethnicity = df_test.groupby('Ethnicity')['y_pred'].mean().reset_index().rename(columns={'y_pred':'Hiring_Rate'})
hiring_rate_age = df_test.groupby('Age_Group')['y_pred'].mean().reset_index().rename(columns={'y_pred':'Hiring_Rate'})

print("\nHiring Rates by Gender:\n", hiring_rate_gender)
print("\nHiring Rates by Ethnicity:\n", hiring_rate_ethnicity)
print("\nHiring Rates by Age Group:\n", hiring_rate_age)


plt.figure(figsize=(8,6))
sns.barplot(x='Gender', y='F1', data=metrics_gender)
plt.title("F1 Score by Gender")
plt.ylim(0,1)
plt.show()


plt.figure(figsize=(10,6))
sns.barplot(x='Ethnicity', y='F1', data=metrics_ethnicity)
plt.title("F1 Score by Ethnicity")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8,6))
sns.barplot(x='Age_Group', y='F1', data=metrics_age)
plt.title("F1 Score by Age Group")
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x='Gender', y='Hiring_Rate', data=hiring_rate_gender)
plt.title("Hiring Rate by Gender")
plt.ylim(0,1)
plt.ylabel("Proportion Hired")
plt.show()


plt.figure(figsize=(10,6))
sns.barplot(x='Ethnicity', y='Hiring_Rate', data=hiring_rate_ethnicity)
plt.title("Hiring Rate by Ethnicity")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.ylabel("Proportion Hired")
plt.show()


plt.figure(figsize=(8,6))
sns.barplot(x='Age_Group', y='Hiring_Rate', data=hiring_rate_age)
plt.title("Hiring Rate by Age Group")
plt.ylim(0,1)
plt.ylabel("Proportion Hired")
plt.show()


def plot_confusion_matrix_for_group(df, group_column, group_value):
    group_df = df[df[group_column] == group_value]
    cm = confusion_matrix(group_df['y_true'], group_df['y_pred'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {group_column} = {group_value}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


for gender in df_test['Gender'].unique():
    plot_confusion_matrix_for_group(df_test, 'Gender', gender)


plt.figure(figsize=(8,6))
sns.kdeplot(data=df_test, x='y_proba', hue='Gender', fill=True)
plt.title("Density of Predicted Probabilities by Gender")
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(data=df_test, x='y_proba', hue='Ethnicity', fill=True)
plt.title("Density of Predicted Probabilities by Ethnicity")
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(data=df_test, x='y_proba', hue='Age_Group', fill=True)
plt.title("Density of Predicted Probabilities by Age Group")
plt.show()


metrics_gender.to_csv("eda_metrics_gender.csv", index=False)
metrics_ethnicity.to_csv("eda_metrics_ethnicity.csv", index=False)
metrics_age.to_csv("eda_metrics_age.csv", index=False)
hiring_rate_gender.to_csv("eda_hiring_rate_gender.csv", index=False)
hiring_rate_ethnicity.to_csv("eda_hiring_rate_ethnicity.csv", index=False)
hiring_rate_age.to_csv("eda_hiring_rate_age.csv", index=False)


if 'Age_Group' not in df_test.columns:
    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
    df_test['Age_Group'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels)


gender_hiring = df_test.groupby('Gender')['y_pred'].mean() * 100


ethnicity_hiring = df_test.groupby('Ethnicity')['y_pred'].mean() * 100


age_hiring = df_test.groupby('Age_Group')['y_pred'].mean() * 100


print("Hiring Percentage by Gender:")
print(gender_hiring)
print("\nHiring Percentage by Ethnicity:")
print(ethnicity_hiring)
print("\nHiring Percentage by Age Group:")
print(age_hiring)


plt.figure(figsize=(8,6))
sns.barplot(x=gender_hiring.index, y=gender_hiring.values)
plt.title("Percentage of Each Gender Hired")
plt.ylabel("Percentage Hired (%)")
plt.xlabel("Gender")
plt.ylim(0, 100)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x=ethnicity_hiring.index, y=ethnicity_hiring.values)
plt.title("Percentage of Each Ethnicity Hired")
plt.ylabel("Percentage Hired (%)")
plt.xlabel("Ethnicity")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x=age_hiring.index.astype(str), y=age_hiring.values)
plt.title("Percentage of Each Age Group Hired")
plt.ylabel("Percentage Hired (%)")
plt.xlabel("Age Group")
plt.ylim(0, 100)
plt.show()

df_hired = df_test[df_test['y_pred'] == 1].copy()

if 'Age_Group' not in df_hired.columns:
    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
    df_hired['Age_Group'] = pd.cut(df_hired['Age'], bins=age_bins, labels=age_labels)


gender_counts = df_hired['Gender'].value_counts(normalize=True) * 100
print("Gender composition of hired population (%):")
print(gender_counts)

plt.figure(figsize=(6,6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Gender Composition of Hired Population")
plt.show()

# Ethnicity breakdown.
ethnicity_counts = df_hired['Ethnicity'].value_counts(normalize=True) * 100
print("\nEthnicity composition of hired population (%):")
print(ethnicity_counts)

plt.figure(figsize=(6,6))
plt.pie(ethnicity_counts, labels=ethnicity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Ethnicity Composition of Hired Population")
plt.show()


age_group_counts = df_hired['Age_Group'].value_counts(normalize=True) * 100
print("\nAge Group composition of hired population (%):")
print(age_group_counts)

plt.figure(figsize=(6,6))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Age Group Composition of Hired Population")
plt.show()