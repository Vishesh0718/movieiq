"""
MovieIQ - Predictive Analytics on Film Success
Stage 5: Streamlit Dashboard & Deployment

Run locally:  streamlit run MovieIQ.py
(Run analysis.py and train_model.py once beforehand to generate the
 cleaned dataset, charts, and trained model this app relies on.)
"""

import ast
import json
import pickle

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MovieIQ", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .stMetric { background: #14141c; border-radius: 12px; padding: 14px 10px; }
    div[data-testid="stMetricValue"] { color: #f5c518; }
    .movieiq-hero {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
        padding: 28px 32px; border-radius: 16px; margin-bottom: 18px;
    }
    .movieiq-hero h1 { color: white; margin-bottom: 4px; }
    .movieiq-hero p { color: #eaeaea; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# DATA / MODEL LOADING (cached so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies_clean.csv")
    df["genre_list"] = df["genre_list"].apply(ast.literal_eval)
    return df


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


df = load_data()
bundle = load_model()
model = bundle["model"]
feature_columns = bundle["feature_columns"]
genre_options_all = bundle["genre_options"]
stats_results = load_json("assets/stats_results.json")
model_metrics = load_json("assets/model_metrics.json")

ALL_GENRES = sorted({g for row in df["genre_list"] for g in row})

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="movieiq-hero">
        <h1>🎬 MovieIQ</h1>
        <p>Predictive analytics on film success — explore the data, see the stats, and get a live prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS  (Stage 5.1)
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
selected_genres = st.sidebar.multiselect("Genre", ALL_GENRES, default=[])
min_vote = st.sidebar.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1)
budget_range = st.sidebar.slider(
    "Budget range ($)",
    int(df["budget"].min()), int(df["budget"].max()),
    (int(df["budget"].min()), int(df["budget"].max())),
)
title_search = st.sidebar.text_input("Search title contains…")

filtered = df[
    (df["vote_average"] >= min_vote)
    & (df["budget"].between(*budget_range))
]
if selected_genres:
    filtered = filtered[filtered["genre_list"].apply(lambda gl: any(g in gl for g in selected_genres))]
if title_search:
    filtered = filtered[filtered["title"].str.contains(title_search, case=False, na=False)]

st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} movies")
st.sidebar.download_button(
    "⬇️ Download filtered data (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="movieiq_filtered.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_overview, tab_eda, tab_stats, tab_model, tab_data = st.tabs(
    ["📊 Overview", "🔎 EDA", "🧪 Statistical Tests", "🤖 Model & Prediction", "📄 Data Explorer"]
)

# --- OVERVIEW -----------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies (filtered)", f"{len(filtered):,}")
    c2.metric("Success rate", f"{filtered['success'].mean()*100:.1f}%" if len(filtered) else "—")
    c3.metric("Avg budget", f"${filtered['budget'].mean()/1e6:,.1f}M" if len(filtered) else "—")
    c4.metric("Avg revenue", f"${filtered['revenue'].mean()/1e6:,.1f}M" if len(filtered) else "—")

    st.subheader("Budget vs. Revenue")
    fig = px.scatter(
        filtered, x="budget", y="revenue", color=filtered["success"].map({0: "Fail", 1: "Success"}),
        color_discrete_map={"Fail": "#e74c3c", "Success": "#2ecc71"},
        hover_data=["title", "primary_genre", "vote_average"],
        labels={"budget": "Budget ($)", "revenue": "Revenue ($)", "color": "Outcome"},
        render_mode="svg",
    )
    max_val = max(filtered["budget"].max(), filtered["revenue"].max()) if len(filtered) else 1
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Correlation between budget and revenue across the full dataset: "
        f"**{stats_results['correlation_budget_revenue']}** — bigger budgets trend toward "
        "bigger revenue, but the spread shows budget alone is far from a guarantee."
    )

# --- EDA ------------------------------------------------------------------
with tab_eda:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Most common genres")
        genre_counts = filtered.explode("genre_list")["genre_list"].value_counts().head(10)
        st.plotly_chart(
            px.bar(genre_counts, orientation="h", labels={"value": "Movies", "index": "Genre"},
                   color_discrete_sequence=["#3498db"]).update_yaxes(autorange="reversed"),
            width="stretch",
        )

    with colB:
        st.subheader("Success rate by genre")
        genre_success = filtered.explode("genre_list").groupby("genre_list")["success"].mean().sort_values(ascending=False).head(10)
        st.plotly_chart(
            px.bar(genre_success, orientation="h", labels={"value": "Success rate", "index": "Genre"},
                   color_discrete_sequence=["#2ecc71"]).update_yaxes(autorange="reversed"),
            width="stretch",
        )

    st.subheader("Popularity, runtime & vote average vs. success")
    metric_choice = st.radio("Feature", ["popularity", "runtime", "vote_average"], horizontal=True)
    fig_box = px.box(
        filtered, x=filtered["success"].map({0: "Fail", 1: "Success"}), y=metric_choice,
        color=filtered["success"].map({0: "Fail", 1: "Success"}),
        color_discrete_map={"Fail": "#e74c3c", "Success": "#2ecc71"},
        labels={"x": "Outcome"},
    )
    st.plotly_chart(fig_box, width="stretch")

    st.subheader("Correlation heatmap (numeric features)")
    st.image("assets/correlation_heatmap.png", use_container_width=False, width=560)
    st.caption(
        "Budget and revenue are the only strongly correlated pair (0.76) — expected, since "
        "success is derived directly from them. That's exactly why revenue is excluded as a "
        "model *input* below: using it would leak the answer into the features."
    )

# --- STATS -----------------------------------------------------------------
with tab_stats:
    st.subheader("T-Test — popularity by outcome")
    st.write(
        "**H0:** mean popularity is the same for successful and unsuccessful movies."
    )
    t = stats_results["ttest_popularity"]
    st.metric("p-value", t["p_value"])
    if t["p_value"] < 0.05:
        st.success(f"p = {t['p_value']} < 0.05 → reject H0. Popularity differs significantly between successful and unsuccessful movies.")
    else:
        st.info(f"p = {t['p_value']} ≥ 0.05 → fail to reject H0. No significant difference detected.")

    st.divider()
    st.subheader("Chi-Square Test — genre vs. success")
    st.write("**H0:** a movie's primary genre is independent of whether it succeeds.")
    c = stats_results["chi2_genre_success"]
    st.metric("p-value", c["p_value"])
    if c["p_value"] < 0.05:
        st.success(f"p = {c['p_value']} < 0.05 → reject H0. Genre and success are associated.")
    else:
        st.info(f"p = {c['p_value']} ≥ 0.05 → fail to reject H0. Genre alone doesn't significantly predict success in this dataset.")

    st.divider()
    st.markdown(
        "**What does a p-value mean?** It's the probability of seeing a difference this "
        "large (or larger) if the null hypothesis were actually true. We use the standard "
        "**0.05** threshold: below it, the observed difference is unlikely to be random "
        "chance, so we treat it as statistically significant."
    )

# --- MODEL & PREDICTION -----------------------------------------------------
with tab_model:
    st.subheader("Model performance (Random Forest, held-out test set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{model_metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{model_metrics['precision']*100:.1f}%")
    m3.metric("Recall", f"{model_metrics['recall']*100:.1f}%")

    colX, colY = st.columns(2)
    with colX:
        st.image("assets/confusion_matrix.png", width="stretch")
    with colY:
        st.image("assets/feature_importance.png", width="stretch")

    st.divider()
    st.subheader("🎯 Try it: predict a movie's success")
    st.caption("Enter a hypothetical movie's details and get a live prediction from the trained model.")

    with st.form("predict_form"):
        p1, p2 = st.columns(2)
        with p1:
            in_budget = st.number_input("Budget ($)", min_value=100_000, max_value=500_000_000, value=50_000_000, step=1_000_000)
            in_popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
        with p2:
            in_runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, value=120)
            in_vote = st.slider("Expected average rating (0-10)", 0.0, 10.0, 6.5)
        in_genre = st.selectbox("Primary genre", genre_options_all)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {c: 0 for c in feature_columns}
        row["budget"] = in_budget
        row["popularity"] = in_popularity
        row["runtime"] = in_runtime
        row["vote_average"] = in_vote
        genre_col = f"primary_genre_{in_genre}"
        if genre_col in row:
            row[genre_col] = 1
        X_input = pd.DataFrame([row])[feature_columns]

        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]

        outcome = "✅ Likely SUCCESS" if pred == 1 else "⚠️ Likely NOT successful"
        st.subheader(outcome)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Predicted probability of success (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if pred == 1 else "#e74c3c"},
                "steps": [
                    {"range": [0, 50], "color": "#3a1f1f"},
                    {"range": [50, 100], "color": "#1f3a24"},
                ],
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_gauge, width="stretch")
        st.caption(
            "Prediction is based only on budget, popularity, runtime, rating, and genre — "
            "revenue is intentionally never used as an input, since it defines the label itself."
        )

# --- DATA EXPLORER ------------------------------------------------------
with tab_data:
    st.subheader("Filtered dataset")
    st.dataframe(
        filtered[["title", "primary_genre", "budget", "revenue", "popularity", "runtime", "vote_average", "success"]]
        .sort_values("revenue", ascending=False),
        width="stretch",
        height=520,
    )

st.divider()
with st.expander("📝 Reflection & limitations"):
    st.markdown(
        """
        **How confident would MovieIQ be for a studio?** Moderately — the model captures real
        signal (accuracy well above the majority-class baseline), but success here is defined
        purely as *revenue > budget*, which ignores marketing spend, release timing, and
        franchise effects that studios actually care about.

        **One limitation:** the dataset has no marketing/P&A budget, release date, or cast/crew
        data — all known drivers of box-office outcomes — so the model can only reason from
        budget, popularity, runtime, rating, and genre.

        **One improvement with more time:** add release-date seasonality, franchise/sequel
        flags, and studio identity as features, and compare Random Forest against gradient
        boosting (e.g. XGBoost) for a stronger baseline.
        """
    )
