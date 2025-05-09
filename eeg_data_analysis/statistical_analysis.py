import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, ttest_rel, spearmanr, zscore
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from itertools import combinations

# Load and clean data
df = pd.read_csv("eeg_data_analysis/eeg_bandpower_preprocessed.csv")

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
for band in bands:
    df[f"{band}_mean"] = df[[c for c in df.columns if f"_{band}" in c]].mean(axis=1)

df_clean = df.dropna(subset=["stage"] + [f"{b}_mean" for b in bands]).copy()
df_clean["stage"] = df_clean["stage"].astype(int)

stage_map = {
    1: {"bias": "biased", "transparency": "no"},
    2: {"bias": "biased", "transparency": "yes"},
    3: {"bias": "fair",   "transparency": "no"},
    4: {"bias": "fair",   "transparency": "yes"},
}
df_clean["bias"] = df_clean["stage"].map(lambda x: stage_map[x]["bias"])
df_clean["transparency"] = df_clean["stage"].map(lambda x: stage_map[x]["transparency"])
df_clean["bias"] = df_clean["bias"].astype("category")
df_clean["transparency"] = df_clean["transparency"].astype("category")

# remove outliers
for band in bands:
    df_clean[f"{band}_z"] = zscore(df_clean[f"{band}_mean"])
    df_clean = df_clean[df_clean[f"{band}_z"].abs() < 3]

#  violin plots by stage 
sns.set(style="whitegrid")
for band in bands:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="stage", y=f"{band}_mean", data=df_clean, palette="Set2", hue="stage", legend=False)
    plt.title(f"{band.title()} Bandpower by Stage")
    plt.ylabel("Bandpower")
    plt.xlabel("Stage")
    plt.tight_layout()
    plt.savefig(f"eeg_data_analysis/violin_{band}.png")
    plt.close()

# kruskal and pairwise comparisons
print("\nKruskal-Wallis & Pairwise Comparisons:")
for band in bands:
    print(f"\n{band.upper()}:")
    groups = [df_clean[df_clean["stage"] == s][f"{band}_mean"] for s in sorted(df_clean["stage"].unique())]
    H, p = kruskal(*groups)
    print(f"Kruskal-Wallis H = {H:.3f} | p = {p:.4f}")
    for s1, s2 in combinations(sorted(df_clean["stage"].unique()), 2):
        u, p = mannwhitneyu(
            df_clean[df_clean["stage"] == s1][f"{band}_mean"],
            df_clean[df_clean["stage"] == s2][f"{band}_mean"]
        )
        print(f"  Stage {s1} vs {s2} → U = {u:.1f}, p = {p:.4f}")

# two way anova 
print("\nTwo-Way ANOVA:")
for band in bands:
    model = ols(f"{band}_mean ~ C(bias) * C(transparency)", data=df_clean).fit()
    table = anova_lm(model)
    print(f"\n{band.upper()}:\n", table)

# t-tests (stage 1 and 2)
channels = ["TP9", "AF7", "AF8", "TP10"]
print("\nPaired t-tests (Stage 1 vs 2):")
for band in bands:
    for i, ch in enumerate(channels):
        col = f"ch{i}_{band}"
        if col in df.columns:
            s1 = df[df["stage"] == 1][col].dropna()
            s2 = df[df["stage"] == 2][col].dropna()
            min_len = min(len(s1), len(s2))
            if min_len > 0:
                t, p = ttest_rel(s1[:min_len], s2[:min_len])
                print(f"{ch}-{band}: t = {t:.2f}, p = {p:.4f}")

# trust decision analysis
print("\nTrust Decision Comparisons:")
df_clean["decision_binary"] = df_clean["label"].map({
    "Accept & Trust": 1,
    "Reject & Distrust": 0
})

# barplot average power by decision
for band in bands:
    plt.figure(figsize=(6, 4))
    sns.barplot(x="label", y=f"{band}_mean", data=df_clean, palette="coolwarm", estimator="mean", errorbar="ci")
    plt.title(f"{band.title()} Bandpower by Trust Decision")
    plt.ylabel("Bandpower")
    plt.xlabel("Decision")
    plt.tight_layout()
    plt.savefig(f"eeg_data_analysis/bar_decision_{band}.png")
    plt.close()

# mann–Whitney U Tests
for band in bands:
    acc = df_clean[df_clean["decision_binary"] == 1][f"{band}_mean"]
    rej = df_clean[df_clean["decision_binary"] == 0][f"{band}_mean"]
    u, p = mannwhitneyu(acc, rej)
    print(f"{band.title()} → U = {u:.1f}, p = {p:.4f}")

# correlation bandpower vs trust rate
print("\nCorrelation with Trust Rate:")
summary = df_clean.groupby("participant").agg({
    "decision_binary": "mean",
    **{f"{band}_mean": "mean" for band in bands}
}).rename(columns={"decision_binary": "trust_rate"}).dropna()

for band in bands:
    rho, pval = spearmanr(summary["trust_rate"], summary[f"{band}_mean"])
    print(f"{band.title()} vs Trust Rate: ρ = {rho:.2f}, p = {pval:.4f}")
    
    # Scatter plot
    plt.figure(figsize=(6, 4))
    sns.regplot(x="trust_rate", y=f"{band}_mean", data=summary)
    plt.title(f"{band.title()} vs Trust Rate")
    plt.xlabel("Trust Rate")
    plt.ylabel(f"{band.title()} Power")
    plt.tight_layout()
    plt.savefig(f"eeg_data_analysis/trust_corr_{band}.png")
    plt.close()
