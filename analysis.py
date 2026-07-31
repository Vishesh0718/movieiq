"""
MovieIQ - Stage 1-3: Data Preparation, EDA, and Statistical Testing
Run this once to generate all charts (assets/) and a clean dataset (movies_clean.csv)
that the Streamlit app and the model-training script both reuse.

Run:  python analysis.py
"""

import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
ASSETS = "assets"

# ---------------------------------------------------------------------------
# STAGE 1 · DATA PREPARATION
# ---------------------------------------------------------------------------
df = pd.read_csv("movies.csv")
print(f"[Stage 1.1] Rows x Cols: {df.shape}")
print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())

# 1.2 zeros/missing check -> a budget or revenue of 0 is not a real movie
# economics figure (usually means "unknown", not "spent/earned nothing"), so
# rows with either value at 0 would corrupt the success label. We drop them.
zero_budget = (df["budget"] <= 0).sum()
zero_revenue = (df["revenue"] <= 0).sum()
print(f"[Stage 1.2] zero/negative budget: {zero_budget}, zero/negative revenue: {zero_revenue}")
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

# 1.3 target label
df["success"] = (df["revenue"] > df["budget"]).astype(int)
balance = df["success"].value_counts(normalize=True)
print(f"[Stage 1.3] success rate: {balance.to_dict()}")

# 1.4 genres: stored as a stringified list of dicts -> extract a clean list
# of genre names plus a single "primary_genre" column for easy filtering/plots.
def parse_genres(cell):
    try:
        items = ast.literal_eval(cell)
        return [d["name"] for d in items]
    except (ValueError, SyntaxError):
        return []

df["genre_list"] = df["genres"].apply(parse_genres)
df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")

df.to_csv("movies_clean.csv", index=False)
print("[Stage 1.4] Saved movies_clean.csv with genre_list / primary_genre columns")

# ---------------------------------------------------------------------------
# STAGE 2 · EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------

# 2.1 Budget vs Revenue scatter
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, x="budget", y="revenue", hue="success",
    palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.6,
)
lims = [0, max(df["budget"].max(), df["revenue"].max())]
plt.plot(lims, lims, "--", color="gray", linewidth=1, label="break-even line")
plt.xlabel("Budget ($)")
plt.ylabel("Revenue ($)")
plt.title("Budget vs Revenue")
plt.legend(title="Success")
plt.tight_layout()
plt.savefig(f"{ASSETS}/budget_vs_revenue.png", dpi=140)
plt.close()

corr_br = df["budget"].corr(df["revenue"])
print(f"[Stage 2.1] correlation(budget, revenue) = {corr_br:.3f}")

# 2.2 Genre trends: frequency + success rate
genre_exploded = df.explode("genre_list")
genre_counts = genre_exploded["genre_list"].value_counts()
genre_success = genre_exploded.groupby("genre_list")["success"].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
genre_counts.head(10).plot(kind="barh", ax=axes[0], color="#3498db")
axes[0].invert_yaxis()
axes[0].set_title("Most Common Genres")
axes[0].set_xlabel("Number of Movies")

genre_success.head(10).plot(kind="barh", ax=axes[1], color="#2ecc71")
axes[1].invert_yaxis()
axes[1].set_title("Highest Success Rate by Genre")
axes[1].set_xlabel("Success Rate")
plt.tight_layout()
plt.savefig(f"{ASSETS}/genre_trends.png", dpi=140)
plt.close()

# 2.3 popularity / runtime / vote_average vs success
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
    sns.boxplot(data=df, x="success", y=col, hue="success", ax=ax,
                palette={0: "#e74c3c", 1: "#2ecc71"}, legend=False)
    ax.set_xticklabels(["Fail", "Success"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_vs_success_boxplots.png", dpi=140)
plt.close()

means_by_success = df.groupby("success")[["popularity", "runtime", "vote_average"]].mean()
print("[Stage 2.3] means by success class:\n", means_by_success)

# 2.4 correlation heatmap
plt.figure(figsize=(6.5, 5.5))
num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{ASSETS}/correlation_heatmap.png", dpi=140)
plt.close()
print("[Stage 2.4] correlation matrix:\n", corr)

# ---------------------------------------------------------------------------
# STAGE 3 · STATISTICAL TESTING
# ---------------------------------------------------------------------------

# 3.1 T-test: does popularity differ between successful and unsuccessful movies?
# H0: mean popularity of successful movies == mean popularity of unsuccessful movies
succ_pop = df.loc[df["success"] == 1, "popularity"]
fail_pop = df.loc[df["success"] == 0, "popularity"]
t_stat, p_val_t = stats.ttest_ind(succ_pop, fail_pop, equal_var=False)
print(f"[Stage 3.1] T-test popularity: t={t_stat:.3f}, p={p_val_t:.4g}")

# 3.2 Chi-square: is primary_genre associated with success?
# H0: primary_genre and success are independent
contingency = pd.crosstab(df["primary_genre"], df["success"])
chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"[Stage 3.2] Chi-square genre~success: chi2={chi2:.3f}, dof={dof}, p={p_val_chi2:.4g}")

# 3.3 save a small JSON so the Streamlit app can display these results
# without re-running the tests
stats_results = {
    "correlation_budget_revenue": round(float(corr_br), 4),
    "ttest_popularity": {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_val_t), 6)},
    "chi2_genre_success": {"chi2": round(float(chi2), 4), "dof": int(dof), "p_value": round(float(p_val_chi2), 6)},
    "success_rate": round(float(balance.get(1, 0.0)), 4),
    "n_rows_after_cleaning": int(len(df)),
    "n_rows_dropped": int(zero_budget + zero_revenue),
}
with open(f"{ASSETS}/stats_results.json", "w") as f:
    json.dump(stats_results, f, indent=2)

print("\nDone. Charts saved to assets/, cleaned data saved to movies_clean.csv,")
print("stats saved to assets/stats_results.json")
