# MovieIQ — Predictive Analytics on Film Success

A complete, ready-to-run implementation of the MovieIQ assignment, built on your
`movies.csv` (2,000 rows: budget, revenue, popularity, runtime, vote_average, title, genres).

**Success rule used throughout:** `success = 1` when `revenue > budget`, else `0`.

## What's in this folder

| File | Purpose |
|---|---|
| `movies.csv` | your raw dataset |
| `analysis.py` | Stages 1–3: cleans data, runs EDA, runs T-test + Chi-square, saves charts |
| `train_model.py` | Stage 4: trains & evaluates the Random Forest, saves the model |
| `MovieIQ.py` | Stage 5: the Streamlit dashboard (main app) |
| `requirements.txt` | all Python dependencies |
| `assets/` | every chart + saved stats/metrics (auto-generated) |
| `movies_clean.csv` | cleaned dataset with `success`, `genre_list`, `primary_genre` (auto-generated) |
| `model.pkl` | trained model bundle (auto-generated) |

## Step-by-step: how to run this yourself

### 1. Set up your environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the data prep + EDA + statistical tests (Stages 1–3)
```bash
python analysis.py
```
This will:
- load `movies.csv`, print row/column counts and summary stats
- check for zero/missing budget or revenue (a `0` almost always means "unknown", not
  "spent/earned nothing" — those rows are dropped so they don't corrupt the success label)
- create the `success` column and print the class balance (**80.7% success / 19.3% fail** —
  imbalanced, which is why the model below uses `class_weight="balanced"`)
- parse the `genres` column (a stringified list of dicts) into a clean `genre_list` and a
  `primary_genre` column for filtering/plots
- save 5 charts to `assets/` (budget vs revenue, genre trends, feature boxplots, correlation
  heatmap) and a cleaned dataset to `movies_clean.csv`
- run a T-test on popularity and a Chi-square test on genre, saving results to
  `assets/stats_results.json`

### 3. Train the model (Stage 4)
```bash
python train_model.py
```
This will:
- build features from `budget`, `popularity`, `runtime`, `vote_average`, and one-hot encoded
  `primary_genre` — **`revenue` is deliberately excluded** since it's what defines the target
  (using it would leak the answer straight into the model)
- split 80/20 train/test (stratified, so both sets keep the same success ratio)
- train a `RandomForestClassifier` (300 trees; each tree votes and the majority wins — this
  averaging is what makes forests more stable than a single decision tree)
- print accuracy / precision / recall and save a confusion matrix + feature-importance chart
- save the trained model to `model.pkl` for the app to load

### 4. Launch the dashboard (Stage 5)
```bash
streamlit run MovieIQ.py
```
Open the local URL it prints (usually `http://localhost:8501`).

### 5. Deploy it publicly (optional, Stage 5 bonus)
1. Push this whole folder to a GitHub repo (include `movies.csv`, `movies_clean.csv`,
   `model.pkl`, and `assets/` — Streamlit Cloud needs the generated files too, or add a
   `analysis.py && train_model.py` step to your deploy script).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and set the
   main file to `MovieIQ.py`.
3. Nothing else to change for a dataset this size — no secrets, no external APIs.

## What was upgraded vs. a basic version (like the sample app you linked)

- **Tabbed layout** (Overview / EDA / Statistical Tests / Model & Prediction / Data Explorer)
  instead of one long scrolling page — easier to navigate.
- **Interactive Plotly charts** (hover tooltips, zoom) instead of static matplotlib images
  for the main scatter/bar/box plots.
- **Three sidebar filters** — genre, minimum rating, *and* a budget range slider — plus a
  live title search box, all combined.
- **CSV download button** for whatever the user has currently filtered.
- **Prediction gauge** — the model doesn't just say success/fail, it shows a probability
  gauge so the user sees model confidence, not just a binary label.
- **Live stats verdicts** — the T-test and Chi-square tabs auto-render "reject/fail to
  reject H0" in plain language based on the actual p-value, not a hardcoded conclusion.
- **Class imbalance handled explicitly** — the dataset is ~81% successful movies, so the
  Random Forest uses `class_weight="balanced"` rather than silently over-predicting success.
- **Reflection section built into the app itself**, not just the write-up, so a reviewer
  sees the limitations without leaving the dashboard.

## Ideas if you want to push it further

- Swap Random Forest for XGBoost/LightGBM and show a model-comparison tab.
- Add a "similar movies" lookup (nearest neighbors on the feature vector).
- Add release-date/seasonality if you can enrich the dataset (e.g. via TMDb API).
- Add SHAP values for per-prediction explainability instead of just global feature importance.
"# movieiq" 
