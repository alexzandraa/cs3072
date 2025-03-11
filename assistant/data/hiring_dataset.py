import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


df_ons_age = pd.read_csv("assistant/data/ons_age_data.csv")
df_ons_gender = pd.read_csv("assistant/data/ons_gender_data.csv")
df_ons_ethnicity = pd.read_csv("assistant/data/ons_ethnicity_data.csv")


df_ons_age["Population_Estimate"] = df_ons_age["Population_Estimate"].fillna(0)
df_ons_gender["Population_Estimate"] = df_ons_gender["Population_Estimate"].fillna(0)
df_ons_ethnicity["Population_Estimate"] = df_ons_ethnicity["Population_Estimate"].fillna(0)


# Age Distribution
df_ons_age_grouped = df_ons_age.groupby(["Occupation", "Age_Group"], as_index=False).agg({"Population_Estimate": "sum"})
age_probs_derived = {}
for occ, group in df_ons_age_grouped.groupby("Occupation"):
    total = group["Population_Estimate"].sum()
    bracket_probs = {}
    for _, row in group.iterrows():
        bracket = row["Age_Group"]
        bracket_probs[bracket] = row["Population_Estimate"] / total if total > 0 else 0
    age_probs_derived[occ] = bracket_probs

# Gender Distribution
df_ons_gender = df_ons_gender.rename(columns={"Sex": "Gender", "Occupation": "ons_occupation_gender"})
gender_probs_derived = {}
for occ, group in df_ons_gender.groupby("ons_occupation_gender"):
    total = group["Population_Estimate"].sum()
    probs = {}
    for _, row in group.iterrows():
        probs[row["Gender"]] = row["Population_Estimate"] / total if total > 0 else 0
    gender_probs_derived[occ] = probs

# Ethnicity Distribution
df_ons_ethnicity = df_ons_ethnicity.rename(columns={"Ethnicity_Group": "Ethnicity", "Occupation": "ons_occupation_ethnicity"})
def map_to_broad_ethnicity(detailed_ethnicity):
    detailed_ethnicity = detailed_ethnicity.strip()
    if "Mixed" in detailed_ethnicity:
        return "Mixed"
    elif detailed_ethnicity.startswith("White"):
        return "White"
    elif "Asian" in detailed_ethnicity:
        return "Asian"
    elif "Black" in detailed_ethnicity:
        return "Black"
    else:
        return "Other"
df_ons_ethnicity["BroadEthnicity"] = df_ons_ethnicity["Ethnicity"].apply(map_to_broad_ethnicity)
df_ons_ethnicity_grouped = df_ons_ethnicity.groupby(["ons_occupation_ethnicity", "BroadEthnicity"], as_index=False).agg({"Population_Estimate": "sum"})
broad_ethnicity_probs_derived = {}
for occ, group in df_ons_ethnicity_grouped.groupby("ons_occupation_ethnicity"):
    total = group["Population_Estimate"].sum()
    probs = {}
    for _, row in group.iterrows():
        probs[row["BroadEthnicity"]] = row["Population_Estimate"] / total if total > 0 else 0
    broad_ethnicity_probs_derived[occ] = probs

print("\nAggregated Broad Ethnicity probabilities (sample):")
for occ, probs in broad_ethnicity_probs_derived.items():
    s = sum(probs.values())
    print(f"Occupation: {occ} -> {probs} (Sum: {s:.2f})")

#working age percentages
candidate_age_probs = {
    "16 to 19": 0.0315, "20 to 24": 0.0855, "25 to 29": 0.1114,
    "30 to 34": 0.1184, "35 to 39": 0.1131, "40 to 44": 0.1071,
    "45 to 49": 0.1076, "50 to 54": 0.1142, "55 to 59": 0.1020,
    "60 to 64": 0.0681, "65 to 69": 0.0255, "70 to 74": 0.0098,
    "75 and over": 0.0059
}
candidate_gender_probs = {"Male": 0.52, "Female": 0.48}
candidate_ethnicity_probs = {"White": 0.807, "Asian": 0.101, "Black": 0.044, "Mixed": 0.025, "Other": 0.023}


def pick_age_from_bracket(bracket_str):
    """Return a random age within the given bracket."""
    parts = bracket_str.split()
    try:
        low = int(parts[0])
        high = int(parts[-1]) if parts[-1].isdigit() else 85
        return np.random.randint(low, high + 1)
    except:
        return np.random.randint(16, 66)

def generate_skill_score():
    
    return np.random.randint(0, 101)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def safe_log_ratio(tech_prob, base_prob):
    
    tech_prob = min(max(tech_prob, 1e-6), 1 - 1e-6)
    base_prob = min(max(base_prob, 1e-6), 1 - 1e-6)
    return np.log(tech_prob / base_prob)


occupations_list = [
    "1137 Information technology directors",
    "2131 IT project managers",
    "2132 IT managers",
    "2133 IT business analysts, architects and systems designers",
    "2134 Programmers and software development professionals",
    "2135 Cyber security professionals",
    "2136 IT quality and testing professionals",
    "2137 IT network professionals",
    "2139 Information technology professionals n.e.c."
]

# Number of candidates per occupation.
candidates_per_occ = 2000
synthetic_data = []

# Overall_Score = Skill_Score * Combined_Bias_Factor, and using a logistic mapping.

for occ in occupations_list:
   
    tech_age = age_probs_derived.get(occ, candidate_age_probs)
    tech_gender = gender_probs_derived.get(occ, candidate_gender_probs)
    tech_ethnicity = broad_ethnicity_probs_derived.get(occ, candidate_ethnicity_probs)
    
    # Compute log ratio (bias) for each category.
    weight_age_factors = {k: safe_log_ratio(tech_age.get(k, candidate_age_probs[k]), candidate_age_probs[k])
                          for k in candidate_age_probs}
    weight_gender_factors = {k: safe_log_ratio(tech_gender.get(k, candidate_gender_probs[k]), candidate_gender_probs[k])
                             for k in candidate_gender_probs}
    weight_ethnicity_factors = {k: safe_log_ratio(tech_ethnicity.get(k, candidate_ethnicity_probs[k]), candidate_ethnicity_probs[k])
                                for k in candidate_ethnicity_probs}
    
    
    bracket_labels = list(candidate_age_probs.keys())
    bracket_values = np.array(list(candidate_age_probs.values()))
    bracket_values = bracket_values / bracket_values.sum()
    
    for i in range(candidates_per_occ):
        record = {}
        record["Occupation"] = occ
        
        # Select Age Bracket and Age.
        chosen_bracket = np.random.choice(bracket_labels, p=bracket_values)
        record["Age_Bracket"] = chosen_bracket
        record["Age"] = pick_age_from_bracket(chosen_bracket)
        
        # Randomly select Gender.
        chosen_gender = np.random.choice(list(candidate_gender_probs.keys()), p=list(candidate_gender_probs.values()))
        record["Gender"] = chosen_gender
        
        # Randomly select Ethnicity.
        chosen_ethnicity = np.random.choice(list(candidate_ethnicity_probs.keys()), p=list(candidate_ethnicity_probs.values()))
        record["Ethnicity"] = chosen_ethnicity
        
        # Compute individual log bias values.
        bias_age = weight_age_factors.get(chosen_bracket, 0)
        bias_gender = weight_gender_factors.get(chosen_gender, 0)
        bias_ethnicity = weight_ethnicity_factors.get(chosen_ethnicity, 0)
        
        record["Bias_Age"] = bias_age
        record["Bias_Gender"] = bias_gender
        record["Bias_Ethnicity"] = bias_ethnicity
        
        # Average the log biases and exponentiate to get a multiplicative bias factor.
        combined_log_bias = (bias_age + bias_gender + bias_ethnicity) / 3.0
        combined_bias_factor = np.exp(combined_log_bias)
        record["Combined_Bias_Factor"] = combined_bias_factor
        
        # Generate a random skill score.
        skill = generate_skill_score()
        record["Skill_Score"] = skill
        
        # Compute the overall score as the product of skill and combined bias factor.
        overall_score = skill * combined_bias_factor
        record["Overall_Score"] = overall_score
        
        
        logit = 0.1 * overall_score - 5  
        final_prob = sigmoid(logit)
        record["Final_Prob"] = final_prob
        
        
        record["Hired"] = int(np.random.rand() < final_prob)
        
        synthetic_data.append(record)

df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.to_csv("assistant/data/synthetic_candidate_pool.csv", index=False)
print("\nSample of final dataset:")
print(df_synthetic.head())


plt.figure(figsize=(10,6))
sns.boxplot(data=df_synthetic, x="Age_Bracket", y="Bias_Age", order=sorted(df_synthetic['Age_Bracket'].unique(), key=lambda x: int(x.split()[0])))
plt.title("Box Plot of Age Bias (Log Ratio) by Age Bracket")
plt.xlabel("Age Bracket")
plt.ylabel("Bias_Age (log ratio)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,6))
sns.boxplot(data=df_synthetic, x="Gender", y="Bias_Gender")
plt.title("Box Plot of Gender Bias (Log Ratio) by Gender")
plt.xlabel("Gender")
plt.ylabel("Bias_Gender (log ratio)")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df_synthetic, x="Ethnicity", y="Bias_Ethnicity")
plt.title("Box Plot of Ethnicity Bias (Log Ratio) by Ethnicity")
plt.xlabel("Ethnicity")
plt.ylabel("Bias_Ethnicity (log ratio)")
plt.show()


mean_combined_bias_age = df_synthetic.groupby("Age_Bracket")["Combined_Bias_Factor"].mean().reset_index()
mean_combined_bias_gender = df_synthetic.groupby("Gender")["Combined_Bias_Factor"].mean().reset_index()
mean_combined_bias_ethnicity = df_synthetic.groupby("Ethnicity")["Combined_Bias_Factor"].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=mean_combined_bias_age, x="Age_Bracket", y="Combined_Bias_Factor", order=sorted(mean_combined_bias_age["Age_Bracket"], key=lambda x: int(x.split()[0])))
plt.title("Mean Combined Bias Factor by Age Bracket")
plt.xlabel("Age Bracket")
plt.ylabel("Mean Combined Bias Factor")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,6))
sns.barplot(data=mean_combined_bias_gender, x="Gender", y="Combined_Bias_Factor")
plt.title("Mean Combined Bias Factor by Gender")
plt.xlabel("Gender")
plt.ylabel("Mean Combined Bias Factor")
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(data=mean_combined_bias_ethnicity, x="Ethnicity", y="Combined_Bias_Factor")
plt.title("Mean Combined Bias Factor by Ethnicity")
plt.xlabel("Ethnicity")
plt.ylabel("Mean Combined Bias Factor")
plt.show()


hiring_rate_gender = df_synthetic.groupby("Gender")["Hired"].mean().reset_index() * 100
hiring_rate_ethnicity = df_synthetic.groupby("Ethnicity")["Hired"].mean().reset_index() * 100


age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
df_synthetic['Age_Group'] = pd.cut(df_synthetic['Age'], bins=age_bins, labels=age_labels)
hiring_rate_age = df_synthetic.groupby("Age_Group")["Hired"].mean().reset_index()
hiring_rate_age["Hired"] = hiring_rate_age["Hired"] * 100

print("Hiring Rate by Age Group (%):")
print(hiring_rate_age)

plt.figure(figsize=(10,6))
sns.barplot(data=hiring_rate_age, x="Age_Group", y="Hired", order=sorted(hiring_rate_age["Age_Group"], key=lambda x: int(x.split('-')[0]) if '-' in x else 0))
plt.title("Hiring Rate by Age Group (%)")
plt.xlabel("Age Group")
plt.ylabel("Hiring Rate (%)")
plt.ylim(0,100)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,6))
sns.barplot(data=hiring_rate_gender, x="Gender", y="Hired")
plt.title("Hiring Rate by Gender (%)")
plt.xlabel("Gender")
plt.ylabel("Hiring Rate (%)")
plt.ylim(0,100)
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(data=hiring_rate_ethnicity, x="Ethnicity", y="Hired")
plt.title("Hiring Rate by Ethnicity (%)")
plt.xlabel("Ethnicity")
plt.ylabel("Hiring Rate (%)")
plt.ylim(0,100)
plt.show()


hired_counts = df_synthetic['Hired'].value_counts()
hired_percent = df_synthetic['Hired'].value_counts(normalize=True) * 100

print("Overall Counts of Hired vs. Not Hired:")
print(hired_counts)
print("\nOverall Percentage of Hired vs. Not Hired:")
print(hired_percent)

labels = ['Not Hired', 'Hired']  
sizes = [hired_counts.get(0, 0), hired_counts.get(1, 0)]
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Overall Hired vs. Not Hired Distribution")
plt.show()
