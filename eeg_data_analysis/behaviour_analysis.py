import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Paths
DECISIONS_DIR = "participant_data/decisions"
participants = range(1, 11)

# Load all user decisions
all_data = []
for p in participants:
    path = os.path.join(DECISIONS_DIR, f"user_decisions_{p}.csv")
    if not os.path.exists(path):
        print(f"Missing P{p}")
        continue
    df = pd.read_csv(path)
    df["participant"] = p
    all_data.append(df)

# Combine into one DataFrame
df_all = pd.concat(all_data, ignore_index=True)

# Clean and map columns
df_all["stage_num"] = df_all["stage"].str.extract(r"Stage (\d)").astype(int)
df_all["trust_binary"] = df_all["decision"].map({
    "Accept & Trust": 1,
    "Reject & Distrust": 0
})

# Map bias and transparency by stage number
bias_map = {1: "Biased", 2: "Biased", 3: "Fair", 4: "Fair"}
transparency_map = {1: "No", 2: "Yes", 3: "No", 4: "Yes"}
df_all["bias"] = df_all["stage_num"].map(bias_map)
df_all["transparency"] = df_all["stage_num"].map(transparency_map)

# ---------- Trust rate by stage ----------
trust_stage = df_all.groupby(["participant", "stage_num"]).agg(
    trust_rate=("trust_binary", "mean")
).reset_index()

plt.figure(figsize=(8, 5))
sns.boxplot(x="stage_num", y="trust_rate", data=trust_stage, palette="Set2")
sns.swarmplot(x="stage_num", y="trust_rate", data=trust_stage, color=".25")
plt.title("Trust Rate by Stage")
plt.xlabel("Stage")
plt.ylabel("Trust Rate (mean per participant)")
plt.tight_layout()
plt.savefig("eeg_data_analysis/trust_rate_by_stage.png")
plt.show()

# ANOVA by stage
model_stage = ols("trust_rate ~ C(stage_num)", data=trust_stage).fit()
anova_stage = anova_lm(model_stage)
print("\nANOVA — Trust Rate by Stage:")
print(anova_stage)

# ---------- Trust rate by bias/fairness ----------
trust_bias = df_all.groupby(["participant", "bias"]).agg(
    trust_rate=("trust_binary", "mean")
).reset_index()

plt.figure(figsize=(6, 4))
sns.boxplot(x="bias", y="trust_rate", data=trust_bias, palette="Pastel1")
sns.swarmplot(x="bias", y="trust_rate", data=trust_bias, color=".25")
plt.title("Trust Rate by Fairness Condition")
plt.tight_layout()
plt.savefig("eeg_data_analysis/trust_rate_by_bias.png")
plt.show()

# ANOVA by bias
model_bias = ols("trust_rate ~ C(bias)", data=trust_bias).fit()
anova_bias = anova_lm(model_bias)
print("\nANOVA — Trust Rate by Bias:")
print(anova_bias)

# ---------- Trust rate by transparency ----------
trust_transparency = df_all.groupby(["participant", "transparency"]).agg(
    trust_rate=("trust_binary", "mean")
).reset_index()

plt.figure(figsize=(6, 4))
sns.boxplot(x="transparency", y="trust_rate", data=trust_transparency, palette="Pastel2")
sns.swarmplot(x="transparency", y="trust_rate", data=trust_transparency, color=".25")
plt.title("Trust Rate by Transparency")
plt.tight_layout()
plt.savefig("eeg_data_analysis/trust_rate_by_transparency.png")
plt.show()

# ANOVA by transparency
model_trans = ols("trust_rate ~ C(transparency)", data=trust_transparency).fit()
anova_trans = anova_lm(model_trans)
print("\nANOVA — Trust Rate by Transparency:")
print(anova_trans)
