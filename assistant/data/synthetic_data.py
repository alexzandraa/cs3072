import os
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration


# Load ONS data
ons = pd.read_csv('assistant/data/ons_gender_data.csv')
occupation_weights = ons['Population_Estimate'] / ons['Population_Estimate'].sum()

# Set number of synthetic candidates
num_candidates = 10000
np.random.seed(42)


# Generate features


# Gender distribution
genders = np.random.choice(['Male', 'Female'], size=num_candidates, p=[0.5, 0.5])

# Numeric features
experience = np.clip(np.random.normal(loc=5, scale=1.5, size=num_candidates), 0, None)
test_score = np.random.uniform(0, 100, size=num_candidates)
interview_score = np.random.uniform(0, 10, size=num_candidates)


# Occupation Sampling


# Filter female ONS records only
ons_female = ons[ons["Sex"] == "Female"].copy()
occupation_weights = ons_female['Population_Estimate'] / ons_female['Population_Estimate'].sum()

# Sample occupations weighted by frequency
ons_sample = ons_female.sample(n=num_candidates, replace=True, weights=occupation_weights, random_state=42)
occupations = ons_sample['Occupation'].values
female_representation = ons_sample['Percentage_within_Occupation'].values / 100.0


# Apply bias and compute hiring probability

# Logistic regression coefficients
intercept = -3.5
coef_experience = 0.20
coef_test_score = 0.03
coef_interview_score = 0.15

# Bias term (only for female candidates)
bias = np.where(np.char.lower(genders) == 'female', -(1 - female_representation), 0)

# Linear hiring score
linear_score = (
    intercept
    + coef_experience * experience
    + coef_test_score * test_score
    + coef_interview_score * interview_score
    + bias
)

# Convert to probability
hiring_probability = expit(linear_score)
hired = np.random.binomial(1, hiring_probability)


# Create Final Dataframe
synthetic_data = pd.DataFrame({
    'Candidate_ID': np.arange(1, num_candidates + 1),
    'Gender': genders,
    'Occupation': occupations,
    'Female_Representation': female_representation,
    'Experience': experience,
    'Test_Score': test_score,
    'Interview_Score': interview_score,
    'Bias_Term': bias,
    'Hiring_Probability': hiring_probability,
    'Hired': hired
})


# Save to CSV
output_folder = 'assistant/data'
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'synthetic_recruitment_data.csv')
synthetic_data.to_csv(output_path, index=False)


print("Overall hire rate:", synthetic_data['Hired'].mean())
print(synthetic_data.head())
