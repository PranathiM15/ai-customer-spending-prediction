from __future__ import annotations

from pathlib import Path
import io
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from ml_engine import (
    BASE_FEATURES, TARGET, DatasetError, load_and_clean_dataset,
    prepare_customer_input, train_and_evaluate,
)

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "customer_spending_sample.csv"

st.set_page_config(
    page_title="SpendSense | Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Visual identity -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#eaf1ff; --muted:#91a2bd; --panel:#111c30; --line:rgba(147,171,207,.16); --mint:#8cf0c8; --blue:#7cb7ff; --orange:#ffb86b; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background: radial-gradient(circle at 85% 0%, #1b3150 0%, #0a1322 38%, #08111f 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0e1a2d,#0a1424); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
.block-container { padding-top:1.45rem; padding-bottom:3rem; max-width:1440px; }
h1,h2,h3 { color:#f2f6ff !important; letter-spacing:-.04em; }
p,li,label { color:#c7d3e7; }
small { color:var(--muted); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(24,40,64,.94),rgba(14,26,44,.94)); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }
[data-testid="stMetricLabel"] { color:#aebed4 !important; }
[data-testid="stMetricValue"] { color:#f1f6ff !important; font-weight:800; }
[data-testid="stMetricDelta"] { color:var(--mint) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line) !important; border-radius:18px; background:rgba(15,28,47,.54); }
.stButton button, .stDownloadButton button { border-radius:10px; border:1px solid rgba(140,240,200,.42); background:linear-gradient(100deg,#8cf0c8,#91c9ff); color:#092035; font-weight:800; min-height:2.7rem; }
.stButton button:hover, .stDownloadButton button:hover { border-color:#fff; color:#071625; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stFileUploader section { background:#101e33 !important; color:#eaf1ff !important; border-color:var(--line) !important; border-radius:10px !important; }
.stTabs [data-baseweb="tab"] { color:#b6c5dc; }
.stTabs [aria-selected="true"] { color:var(--mint) !important; }
hr { border-color:var(--line); }
.hero { padding:26px 28px; border:1px solid rgba(140,240,200,.22); border-radius:22px; background:linear-gradient(115deg,rgba(28,52,78,.92),rgba(17,31,51,.88)); position:relative; overflow:hidden; margin-bottom:22px; }
.hero:after { content:'✦'; position:absolute; right:34px; top:12px; font-size:110px; color:rgba(140,240,200,.10); }
.eyebrow { font:500 11px 'DM Mono',monospace; color:var(--mint); letter-spacing:.18em; text-transform:uppercase; }
.hero h1 { font-size:36px; margin:8px 0 7px; }
.hero p { max-width:760px; margin:0; color:#bac9df; }
.pill { display:inline-block; padding:5px 10px; border:1px solid rgba(140,240,200,.25); border-radius:50px; color:var(--mint); font:11px 'DM Mono',monospace; background:rgba(140,240,200,.07); margin-right:6px; }
.section-kicker { color:var(--mint); font:500 11px 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; margin-bottom:5px; }
.insight { padding:15px 17px; border:1px solid var(--line); border-left:3px solid var(--mint); border-radius:12px; background:rgba(17,31,51,.75); margin:8px 0; color:#d7e2f3; }
.muted { color:var(--muted); font-size:13px; }
.stDataFrame { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = "CUSTOMER INTELLIGENCE WORKSPACE") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">✦ {tag}</div><h1>{title}</h1>'
        f'<p>{subtitle}</p></div>', unsafe_allow_html=True
    )


def section(label: str, title: str) -> None:
    st.markdown(f'<div class="section-kicker">{label}</div><h3>{title}</h3>', unsafe_allow_html=True)


def style_plot(fig) -> None:
    fig.patch.set_facecolor("#111c30")
    for ax in fig.axes:
        ax.set_facecolor("#111c30")
        ax.tick_params(colors="#b7c7dc", labelsize=9)
        ax.xaxis.label.set_color("#b7c7dc")
        ax.yaxis.label.set_color("#b7c7dc")
        ax.title.set_color("#f2f6ff")
        for spine in ax.spines.values():
            spine.set_color("#2a3a52")
        ax.grid(color="#26364d", alpha=.5, linewidth=.7)
    fig.tight_layout()


@st.cache_data(show_spinner=False)
def get_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_resource(show_spinner="Preparing data and training the models...")
def cached_training(csv_bytes: bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    cleaned, report = load_and_clean_dataset(raw)
    trained = train_and_evaluate(cleaned)
    return cleaned, report, trained


# --- Sidebar navigation ----------------------------------------------------
with st.sidebar:
    st.markdown("<div class='eyebrow'>✦ SPENDSENSE / 01</div>", unsafe_allow_html=True)
    st.markdown("# SpendSense")
    st.caption("Customer spending intelligence")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Explore Data", "Model Studio", "Predict Customer", "Business Brief"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.markdown("<span class='pill'>DATA READY</span><span class='pill'>ML PIPELINE</span>", unsafe_allow_html=True)
    st.caption("Built with Python · Pandas · NumPy · Seaborn · Scikit-learn")
    st.caption("Demo records are synthetic. Upload approved data for your own analysis.")


# --- Dataset selection -----------------------------------------------------
st.sidebar.markdown("### Dataset")
uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"], help="Use the sample file's column names as a template.")
if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    dataset_label = uploaded_file.name
else:
    raw_bytes = SAMPLE_PATH.read_bytes()
    dataset_label = "customer_spending_sample.csv"

try:
    with st.spinner("Preparing customer data and fitting models..."):
        data, cleaning_report, results = cached_training(raw_bytes)
except (DatasetError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Dataset could not be prepared: {exc}")
    st.info("Tip: download the sample CSV from the Overview page and use it as your upload template.")
    st.stop()
except Exception as exc:
    st.error(f"The project hit an unexpected data/model issue: {exc}")
    st.info("Try the bundled sample CSV first. If this continues, check that all required columns are present and contain usable values.")
    st.stop()

metrics = results["metrics"]
best_name = results["best_model_name"]
best_model = results["best_model"]


# --- Overview --------------------------------------------------------------
if page == "Overview":
    hero("Understand the spend behind every customer.",
         "A practical machine-learning workspace to clean customer data, discover spending patterns, compare classification models, and score a new customer.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CUSTOMERS ANALYSED", f"{len(data):,}")
    c2.metric("SPENDING SEGMENTS", f"{data[TARGET].nunique()}")
    c3.metric("TOP MODEL", best_name)
    c4.metric("BEST WEIGHTED F1", f"{metrics.iloc[0]['F1-score']:.1%}")

    left, right = st.columns([1.25, 1])
    with left:
        section("01 / SEGMENT SNAPSHOT", "Where customers sit")
        counts = data[TARGET].value_counts().reindex(["Low", "Medium", "High"]).dropna()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#7cb7ff", "#ffb86b", "#8cf0c8"])
        ax.set_xlabel("Spending category")
        ax.set_ylabel("Customer count")
        ax.set_title("Customer mix by spending category", loc="left", pad=12, fontweight="bold")
        style_plot(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        section("02 / MODEL LEADERBOARD", "Baseline model comparison")
        display_metrics = metrics.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
            display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.1%}")
        st.dataframe(display_metrics, hide_index=True, use_container_width=True)
        st.markdown(f"<div class='insight'><b>Current leader:</b> {best_name}. It achieved the highest weighted F1-score on the held-out test set.</div>", unsafe_allow_html=True)

    section("03 / PROJECT PIPELINE", "From raw CSV to customer prediction")
    steps = st.columns(5)
    for col, num, label, desc in zip(
        steps,
        ["01", "02", "03", "04", "05"],
        ["Clean", "Explore", "Engineer", "Evaluate", "Predict"],
        ["Validate columns, remove duplicates, handle missing values", "Study distributions, relationships and correlations", "Create purchase value and visit efficiency features", "Compare three models with weighted metrics", "Classify a new customer into a spending segment"],
    ):
        with col:
            st.markdown(f"<div class='insight'><div class='eyebrow'>{num}</div><b>{label}</b><p class='muted'>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='muted'>Dataset: " + dataset_label + " · Sample data is synthetic and generated for demonstration.</div>", unsafe_allow_html=True)
    st.download_button("Download cleaned customer dataset", data=data.to_csv(index=False).encode("utf-8"), file_name="cleaned_customer_spending.csv", mime="text/csv")
    st.download_button("Download sample CSV template", data=get_sample_data().to_csv(index=False).encode("utf-8"), file_name="customer_spending_sample.csv", mime="text/csv")


# --- Explore data ----------------------------------------------------------
elif page == "Explore Data":
    hero("Explore the customer landscape.", "Review data quality, spending patterns, purchasing behaviour and relationships before interpreting model results.", "DATA LAB / EXPLORATORY ANALYSIS")
    q1, q2, q3 = st.columns(3)
    q1.metric("ROWS AFTER CLEANING", f"{len(data):,}")
    q2.metric("DUPLICATES REMOVED", f"{cleaning_report['duplicates_removed']:,}")
    q3.metric("MISSING NUMERIC CELLS", f"{cleaning_report['missing_numeric_cells']:,}")

    with st.expander("Data cleaning and preparation log", expanded=False):
        st.json(cleaning_report)
        st.write("Numeric missing values are imputed with the training-set median; gender is imputed with the most frequent value. Annual spend is excluded from model inputs to avoid target leakage.")

    section("01 / DATA PREVIEW", "Cleaned customer records")
    st.dataframe(data.drop(columns=["customer_id"], errors="ignore").head(15), use_container_width=True, hide_index=True)

    section("02 / DISTRIBUTION", "Six views of the spending data")
    col1, col2 = st.columns(2)
    with col1:
        counts = data[TARGET].value_counts().reindex(["Low", "Medium", "High"]).dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=data, x=TARGET, order=["Low", "Medium", "High"], ax=ax, palette=["#7cb7ff", "#ffb86b", "#8cf0c8"])
        ax.set_title("1. Spending category distribution", loc="left", fontweight="bold")
        ax.set_xlabel("Spending category")
        ax.set_ylabel("Number of customers")
        style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        if "annual_spend" in data and data["annual_spend"].notna().any():
            sns.histplot(data=data, x="annual_spend", hue=TARGET, bins=28, element="step", stat="count", common_norm=False, ax=ax)
            ax.set_xlabel("Annual spend (₹)")
            ax.set_title("2. Annual spending distribution", loc="left", fontweight="bold")
        else:
            ax.text(.5, .5, "No annual_spend column supplied", ha="center", color="#b7c7dc")
            ax.set_title("2. Annual spending distribution", loc="left", fontweight="bold")
        style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=data, x=TARGET, y="purchase_frequency", order=["Low", "Medium", "High"], ax=ax, palette=["#7cb7ff", "#ffb86b", "#8cf0c8"])
        ax.set_title("3. Purchase frequency by segment", loc="left", fontweight="bold")
        ax.set_xlabel("Spending category"); ax.set_ylabel("Purchases per month")
        style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=data.sample(min(650, len(data)), random_state=42), x="annual_income", y="avg_order_value", hue=TARGET, alpha=.72, s=35, ax=ax)
        ax.set_title("4. Income vs average order value", loc="left", fontweight="bold")
        ax.set_xlabel("Annual income (₹)"); ax.set_ylabel("Average order value (₹)")
        style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)

    col5, col6 = st.columns(2)
    with col5:
        numeric = data.select_dtypes(include=np.number).drop(columns=["annual_spend"], errors="ignore")
        corr = numeric.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        sns.heatmap(corr, cmap="crest", center=0, linewidths=.35, ax=ax, cbar_kws={"shrink": .75})
        ax.set_title("5. Numeric feature correlation", loc="left", fontweight="bold")
        style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col6:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        sns.barplot(data=data, x="gender", y="purchase_frequency", hue=TARGET, estimator="mean", errorbar=None, ax=ax)
        ax.set_title("6. Average purchase frequency by gender", loc="left", fontweight="bold")
        ax.set_xlabel("Gender group"); ax.set_ylabel("Average purchases per month")
        style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.download_button("Export cleaned data as CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="cleaned_customer_spending.csv", mime="text/csv")


# --- Model Studio ----------------------------------------------------------
elif page == "Model Studio":
    hero("Choose a model with evidence.", "Three classification baselines are trained on the same stratified hold-out split. Compare their metrics, inspect the best model and review the most influential features.", "MODEL LAB / CLASSIFICATION")
    top = metrics.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("BEST MODEL", best_name)
    b.metric("ACCURACY", f"{top['Accuracy']:.1%}")
    c.metric("WEIGHTED PRECISION", f"{top['Precision']:.1%}")
    d.metric("WEIGHTED RECALL", f"{top['Recall']:.1%}")

    section("01 / FAIR COMPARISON", "Model performance")
    pretty = metrics.copy()
    st.dataframe(pretty.style.format({col: "{:.3f}" for col in ["Accuracy", "Precision", "Recall", "F1-score"]}), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    metrics_melt = metrics.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"], var_name="Metric", value_name="Score")
    sns.barplot(data=metrics_melt, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison across evaluation metrics", loc="left", fontweight="bold")
    style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)

    section("02 / ERROR ANALYSIS", "Confusion matrix")
    selected_model = st.selectbox("Select a model to inspect", list(results["models"].keys()), index=list(results["models"].keys()).index(best_name))
    selected_pipeline = results["models"][selected_model]
    predictions = selected_pipeline.predict(results["X_test"])
    labels = results["class_names"]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results["y_test"], predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels, linewidths=.5, ax=ax)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Actual category")
    ax.set_title(f"{selected_model} · held-out test set", loc="left", fontweight="bold")
    style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption(f"Training rows: {results['train_rows']:,} · Test rows: {results['test_rows']:,} · Stratified split: 75% / 25%.")

    section("03 / MODEL INTERPRETATION", "Which customer signals mattered most?")
    importance = results["feature_importance"].head(10).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(importance["feature"], importance["importance"], color="#8cf0c8")
    ax.set_xlabel("Relative importance")
    ax.set_title(f"Feature importance · {best_name}", loc="left", fontweight="bold")
    style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption("Importance is extracted from the selected model (tree impurity importance or absolute logistic-regression coefficients). It indicates model influence, not causation.")


# --- Prediction ------------------------------------------------------------
elif page == "Predict Customer":
    hero("Score a customer in seconds.", "Enter demographic and purchasing signals to estimate whether a customer belongs to the Low, Medium or High spending segment.", "LIVE PREDICTION / NEW CUSTOMER")
    left, right = st.columns([1.05, .95])
    with left:
        section("01 / CUSTOMER PROFILE", "Enter customer details")
        with st.form("customer_prediction_form"):
            f1, f2 = st.columns(2)
            with f1:
                age = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
                gender = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Unknown"])
                annual_income = st.number_input("Annual income (₹)", min_value=0, value=650000, step=10000)
                purchase_frequency = st.number_input("Purchases per month", min_value=0, max_value=100, value=5, step=1)
                avg_order_value = st.number_input("Average order value (₹)", min_value=0, value=2400, step=100)
            with f2:
                website_visits = st.number_input("Website visits per month", min_value=0, max_value=1000, value=18, step=1)
                loyalty_years = st.number_input("Customer loyalty (years)", min_value=0.0, max_value=60.0, value=2.5, step=0.5)
                discount_usage = st.slider("Discount usage rate", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
                total_purchases = st.number_input("Total purchases to date", min_value=0, value=90, step=1)
            submitted = st.form_submit_button("Predict spending category", use_container_width=True)

    with right:
        section("02 / PREDICTION RESULT", "Model output")
        if submitted:
            if annual_income <= 0 or avg_order_value <= 0 or purchase_frequency <= 0:
                st.warning("Enter a positive annual income, average order value and purchase frequency before predicting.")
            else:
                values = {
                    "age": age, "gender": gender, "annual_income": annual_income,
                    "purchase_frequency": purchase_frequency, "avg_order_value": avg_order_value,
                    "website_visits": website_visits, "loyalty_years": loyalty_years,
                    "discount_usage": discount_usage, "total_purchases": total_purchases,
                }
                one_customer = prepare_customer_input(values)
                predicted = best_model.predict(one_customer)[0]
                st.markdown(f"<div class='insight'><div class='eyebrow'>PREDICTED SEGMENT</div><h1 style='font-size:38px;margin:7px 0'>{predicted} spender</h1><p class='muted'>Predicted by {best_name}</p></div>", unsafe_allow_html=True)
                if hasattr(best_model, "predict_proba"):
                    probability = best_model.predict_proba(one_customer)[0]
                    probability_df = pd.DataFrame({"Segment": best_model.classes_, "Probability": probability}).sort_values("Probability", ascending=False)
                    fig, ax = plt.subplots(figsize=(6.5, 2.8))
                    sns.barplot(data=probability_df, x="Segment", y="Probability", ax=ax, palette=["#7cb7ff", "#ffb86b", "#8cf0c8"])
                    ax.set_ylim(0, 1); ax.set_title("Prediction confidence by segment", loc="left", fontweight="bold")
                    style_plot(fig); st.pyplot(fig, use_container_width=True); plt.close(fig)
                    st.dataframe(probability_df.style.format({"Probability": "{:.1%}"}), hide_index=True, use_container_width=True)
                estimated_monthly = purchase_frequency * avg_order_value
                st.markdown(f"<div class='insight'><b>Derived purchase value:</b> ₹{estimated_monthly:,.0f} per month, based on the inputs provided.</div>", unsafe_allow_html=True)
                st.download_button("Download this prediction", data=pd.DataFrame([values | {"predicted_category": predicted}]).to_csv(index=False).encode("utf-8"), file_name="customer_prediction.csv", mime="text/csv")
        else:
            st.markdown("<div class='insight'><b>Ready when you are.</b><p class='muted'>Complete the customer profile and run the prediction to see the segment and class probabilities here.</p></div>", unsafe_allow_html=True)
            st.markdown("**Model used:** " + best_name)
            st.markdown("**Input signals:** demographics, income, purchase frequency, basket value, digital activity, loyalty and discount usage.")


# --- Business brief --------------------------------------------------------
else:
    hero("Turn model output into business action.", "A compact summary of the dataset, spending mix, model quality and practical next steps for customer engagement.", "BUSINESS BRIEF / DECISION SUPPORT")
    counts = data[TARGET].value_counts()
    total = max(len(data), 1)
    share = {key: counts.get(key, 0) / total for key in ["Low", "Medium", "High"]}
    avg_spend = data["annual_spend"].mean() if "annual_spend" in data and data["annual_spend"].notna().any() else np.nan
    f1 = float(metrics.iloc[0]["F1-score"])
    st.markdown("### Executive snapshot")
    spend_text = f"₹{avg_spend:,.0f}" if pd.notna(avg_spend) else "not provided"
    st.markdown(
        f"The analysis covers **{len(data):,} customer records**. The segment mix is "
        f"**{share['Low']:.1%} Low**, **{share['Medium']:.1%} Medium**, and **{share['High']:.1%} High**. "
        f"Average annual spend is **{spend_text}**. The leading classifier is **{best_name}**, "
        f"with a weighted F1-score of **{f1:.1%}** on the held-out test set."
    )

    section("01 / ACTIONABLE INTERPRETATION", "Suggested customer strategies")
    st.markdown(f"<div class='insight'><b>High-spending customers — {share['High']:.1%} of the data</b><br/>Prioritise retention, loyalty benefits and relevant premium recommendations. Monitor repeat purchase behaviour rather than relying only on one large order.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='insight'><b>Medium-spending customers — {share['Medium']:.1%} of the data</b><br/>Test product bundles, useful add-ons and personalised offers that encourage repeat purchases while protecting margin.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='insight'><b>Low-spending customers — {share['Low']:.1%} of the data</b><br/>Use helpful onboarding, discovery campaigns and low-friction offers. Avoid assuming low potential from one short observation period.</div>", unsafe_allow_html=True)

    section("02 / MODEL CAVEATS", "How to interpret the results")
    st.markdown("The reported scores measure performance on a held-out split of this dataset; they are not a guarantee of future business performance. The bundled dataset is synthetic. Before business use, train and validate on representative, consented customer data, check for bias and drift, and confirm that the spending-category definition matches the organisation's policy. Feature importance shows association within the fitted model, not causal effects.")

    report_text = f"""SPENDSENSE — CUSTOMER SPENDING PREDICTION BUSINESS BRIEF

Dataset: {dataset_label}
Customer records after cleaning: {len(data)}
Duplicate rows removed: {cleaning_report['duplicates_removed']}
Invalid target rows removed: {cleaning_report['invalid_target_rows_removed']}

SPENDING MIX
Low: {share['Low']:.1%}
Medium: {share['Medium']:.1%}
High: {share['High']:.1%}
Average annual spend: {spend_text}

MODEL EVALUATION (weighted)
{metrics.to_string(index=False)}
Best model: {best_name}
Training rows: {results['train_rows']}
Test rows: {results['test_rows']}

BUSINESS SUGGESTIONS
- High: prioritise retention and relevant premium recommendations.
- Medium: test useful bundles and personalised add-ons.
- Low: focus on discovery, onboarding and low-friction engagement.

LIMITATIONS
The sample dataset is synthetic. Scores are hold-out estimates, not guarantees. Validate on representative real data before operational decisions. Feature importance is not causation.
"""
    st.download_button("Download business insight report (.txt)", data=report_text, file_name="customer_spending_business_brief.txt", mime="text/plain")
    st.download_button("Download model comparison (.csv)", data=metrics.to_csv(index=False).encode("utf-8"), file_name="model_comparison_metrics.csv", mime="text/csv")

st.divider()
st.caption("SPENDSENSE · AI-BASED CUSTOMER SPENDING PREDICTION · Local demo application · Built for learning and project evaluation")
