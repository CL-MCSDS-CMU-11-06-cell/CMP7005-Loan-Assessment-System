from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
MODELS = BASE / "models"

st.set_page_config(
    page_title="Retail Bank Loan System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# Custom blue / green / white GUI styling
# -------------------------------------------------------
st.markdown(
    """
    <style>
        /* Hide Streamlit's default top decoration / sidebar controls */
        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="collapsedControl"] {
            display: none;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Main page */
        .stApp {
            background:
                linear-gradient(180deg, #eef8ff 0%, #ffffff 38%, #f2fbf7 100%);
            color: #17324d;
            font-family: "Segoe UI", Arial, sans-serif;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        /* Headings */
        h1, h2, h3 {
            font-family: "Trebuchet MS", "Segoe UI", sans-serif !important;
            color: #0c4a6e !important;
            letter-spacing: -0.02em;
        }

        h1 {
            font-weight: 800 !important;
        }

        p, label, div, span {
            font-family: "Segoe UI", Arial, sans-serif;
        }

        /* Hero/header */
        .bank-header {
            background: linear-gradient(115deg, #0b5ea8 0%, #0a8f78 100%);
            border-radius: 18px;
            padding: 24px 30px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(12, 74, 110, 0.16);
        }

        .bank-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.05rem;
        }

        .bank-header p {
            color: #e9fbff;
            margin: 8px 0 0 0;
            font-size: 1rem;
        }

        /* Top navigation tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: white;
            border: 1px solid #c9e8e0;
            border-radius: 15px;
            padding: 6px;
            gap: 5px;
            box-shadow: 0 5px 18px rgba(11, 94, 168, 0.08);
            position: sticky;
            top: 0.35rem;
            z-index: 999;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 11px;
            color: #0c4a6e;
            font-weight: 650;
            padding-left: 14px;
            padding-right: 14px;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(100deg, #0b5ea8, #0a8f78) !important;
            color: white !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #d8eee8;
            padding: 16px 18px;
            border-radius: 14px;
            box-shadow: 0 5px 16px rgba(12, 74, 110, 0.07);
        }

        [data-testid="stMetricLabel"] {
            color: #41637e;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #0a8f78;
            font-weight: 800;
        }

        /* Inputs */
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            border-radius: 10px !important;
        }

        /* Buttons */
        .stButton > button,
        .stFormSubmitButton > button {
            border: none;
            border-radius: 10px;
            background: linear-gradient(100deg, #0b5ea8, #0a8f78);
            color: white;
            font-weight: 700;
            padding: 0.55rem 1.15rem;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: linear-gradient(100deg, #084b86, #087963);
            color: white;
            border: none;
        }

        /* Dataframes / info boxes */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #d8eee8;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Soft section card */
        .soft-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #d5ece6;
            border-radius: 14px;
            padding: 18px 20px;
            margin: 10px 0 18px 0;
            box-shadow: 0 5px 16px rgba(12, 74, 110, 0.05);
        }

        .flow-box {
            background: linear-gradient(90deg, #e8f5ff, #e9faf4);
            border-left: 5px solid #0a8f78;
            border-radius: 12px;
            padding: 16px 18px;
            color: #17324d;
            font-weight: 600;
        }

        /* Reduce excessive tab content top spacing */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv(OUTPUT / "cleaned_loan_data.csv")

@st.cache_data
def load_json(name):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))

@st.cache_resource
def load_loan_model():
    return joblib.load(MODELS / "loan_approval_model.joblib")

@st.cache_resource
def load_cluster_model():
    return joblib.load(MODELS / "kmeans_customer_segmentation.joblib")

data = load_data()
metadata = load_json("metadata.json")
metrics = load_json("model_metrics.json")
loan_model = load_loan_model()
cluster_model = load_cluster_model()

# -------------------------------------------------------
# Main header
# -------------------------------------------------------
st.markdown(
    """
    <div class="bank-header">
        <h1>🏦 Retail Bank Loan Assessment System</h1>
        <p>CMP7005 • Data preprocessing, EDA, prediction and customer segmentation</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# TOP NAVIGATION — tabs instead of radio buttons
# -------------------------------------------------------
tab_dashboard, tab_clean, tab_eda, tab_predict, tab_segment, tab_performance = st.tabs(
    [
        "Dashboard",
        "Cleaned Data",
        "EDA",
        "Loan Prediction",
        "Segmentation",
        "Performance"
    ]
)

# =======================================================
# Dashboard
# =======================================================
with tab_dashboard:
    st.subheader("Dashboard")
    st.write(
        "This application presents the results produced by the Google Colab notebook "
        "in a simple bank-style interface."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(data):,}")
    c2.metric("Approved", f"{int(data['LoanApproved'].sum()):,}")
    c3.metric("Approval Rate", f"{data['LoanApproved'].mean()*100:.1f}%")
    c4.metric("Average Loan", f"${data['LoanAmount'].mean():,.0f}")

    st.markdown("### Project Flow")
    st.markdown(
        """
        <div class="flow-box">
        Two raw CSV files → Cleaning → Merge using ID → Feature Engineering →
        EDA → Logistic Regression → K-Means → Streamlit App
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### What the system does")
    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="soft-card">
            <b>Data side</b><br><br>
            • Cleans invalid and missing values<br>
            • Merges both loan datasets using ID<br>
            • Creates monthly income, loan payment and DTI<br>
            • Shows EDA and model performance
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="soft-card">
            <b>Banking side</b><br><br>
            • Estimates historical loan approval probability<br>
            • Groups customers using K-Means<br>
            • Shows risk-oriented customer segments<br>
            • Supports a simple decision-support demonstration
            </div>
            """,
            unsafe_allow_html=True
        )

# =======================================================
# Cleaned Data
# =======================================================
with tab_clean:
    st.subheader("Cleaned and Preprocessed Data")

    c1, c2 = st.columns(2)
    c1.metric("RiskScore Median Used", f"{metadata['risk_median']:.0f}")
    c2.metric("MaritalStatus Mode", metadata["marital_mode"])

    st.markdown(
        """
        <div class="soft-card">
        <b>Main cleaning decisions</b><br><br>
        • Invalid RiskScore values such as <code>XX</code> were converted to missing values and filled using the median.<br>
        • Missing Experience values in Dataset 2 were recovered from Dataset 1 using the matching ID.<br>
        • Missing MaritalStatus values were filled using the most frequent value.<br>
        • Dataset 1 ApplicationDate was used as the consistent date source.
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = [
        "ID", "AnnualIncome", "Monthlyincome", "LoanAmount",
        "Monthlyloanpyament", "MonthlyDebtPayments",
        "Totaldebttoincomeratio", "RiskScore", "LoanApproved"
    ]

    st.markdown("### Data Preview")
    st.dataframe(
        data[cols].head(200),
        use_container_width=True,
        hide_index=True
    )

# =======================================================
# EDA
# =======================================================
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    variable = st.selectbox(
        "Select a variable for distribution",
        [
            "AnnualIncome",
            "LoanAmount",
            "CreditScore",
            "RiskScore",
            "Totaldebttoincomeratio"
        ],
        key="eda_distribution"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data[variable].dropna(), bins=30)
    ax.set_title(f"Distribution of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown("### Approved vs Not Approved")
    compare = st.selectbox(
        "Select a variable to compare",
        [
            "CreditScore",
            "RiskScore",
            "AnnualIncome",
            "LoanAmount",
            "Totaldebttoincomeratio"
        ],
        key="eda_compare"
    )

    summary = (
        data.groupby("LoanApproved")[compare]
        .mean()
        .rename(index={0: "Not Approved", 1: "Approved"})
    )
    st.bar_chart(summary)

    st.markdown("### Approval Rate by Employment Status")
    employment_rate = (
        data.groupby("EmploymentStatus")["LoanApproved"]
        .mean()
        .mul(100)
    )
    st.bar_chart(employment_rate)

# =======================================================
# Loan Prediction
# =======================================================
with tab_predict:
    st.subheader("Loan Approval Prediction")
    st.write(
        "Enter applicant details below. The model returns the probability of the "
        "historical `LoanApproved` class."
    )

    with st.form("loan_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age",
                18, 90,
                int(data["Age"].median())
            )
            credit = st.number_input(
                "Credit Score",
                300, 850,
                int(data["CreditScore"].median())
            )
            income = st.number_input(
                "Annual Income",
                1000.0, 1000000.0,
                float(data["AnnualIncome"].median()),
                step=1000.0
            )
            employment = st.selectbox(
                "Employment Status",
                metadata["employment_values"]
            )
            experience = st.number_input(
                "Experience",
                0.0, 70.0,
                float(data["Experience"].median())
            )
            risk = st.number_input(
                "Risk Score",
                0.0, 100.0,
                float(data["RiskScore"].median())
            )

        with c2:
            loan_amount = st.number_input(
                "Loan Amount",
                500.0, 500000.0,
                float(data["LoanAmount"].median()),
                step=500.0
            )
            duration = st.selectbox(
                "Loan Duration (months)",
                metadata["loan_durations"]
            )
            debt = st.number_input(
                "Monthly Debt Payments",
                0.0, 10000.0,
                float(data["MonthlyDebtPayments"].median()),
                step=50.0
            )
            defaults = st.selectbox(
                "Previous Loan Defaults",
                [0, 1]
            )
            payment_history = st.number_input(
                "Payment History",
                int(data["PaymentHistory"].min()),
                int(data["PaymentHistory"].max()),
                int(data["PaymentHistory"].median())
            )
            interest = st.number_input(
                "Interest Rate (%)",
                0.0, 60.0,
                float(data["InterestRate"].median()),
                step=0.1
            )

            # Selectbox instead of radio button
            mode = st.selectbox(
                "Prediction Mode",
                ["Improved", "Baseline"],
                help="Improved uses the validation-selected threshold."
            )

        submitted = st.form_submit_button(
            "Assess Application",
            type="primary"
        )

    if submitted:
        monthly_income = income / 12.0
        monthly_rate = interest / 100.0 / 12.0

        if monthly_rate == 0:
            monthly_payment = loan_amount / duration
        else:
            monthly_payment = (
                loan_amount
                * (monthly_rate * (1 + monthly_rate) ** duration)
                / ((1 + monthly_rate) ** duration - 1)
            )

        dti = (
            (monthly_payment + debt)
            * 100.0
            / monthly_income
        )

        applicant = pd.DataFrame([{
            "Age": age,
            "CreditScore": credit,
            "AnnualIncome": income,
            "EmploymentStatus": employment,
            "Experience": experience,
            "LoanAmount": loan_amount,
            "LoanDuration": duration,
            "MonthlyDebtPayments": debt,
            "PreviousLoanDefaults": defaults,
            "PaymentHistory": payment_history,
            "InterestRate": interest,
            "RiskScore": risk,
            "Totaldebttoincomeratio": dti
        }])

        probability = float(
            loan_model.predict_proba(applicant)[0, 1]
        )

        threshold = (
            metrics["improved_threshold"]
            if mode == "Improved"
            else metrics["baseline_threshold"]
        )

        prediction = int(
            probability >= threshold
        )

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Approval Probability",
            f"{probability*100:.1f}%"
        )
        r2.metric(
            "Calculated DTI",
            f"{dti:.1f}%"
        )
        r3.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

        if prediction == 1:
            st.success("Model result: APPROVAL")
        else:
            st.warning("Model result: NON-APPROVAL")



# =======================================================
# Customer Segmentation
# =======================================================
with tab_segment:
    st.subheader("Customer Segmentation")

    profile = pd.read_csv(
        OUTPUT / "cluster_profile.csv"
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="soft-card">
        <b>K-Means segmentation</b><br><br>
        Customers are grouped into three data-driven clusters.
        The labels <b>High Risk</b>, <b>Moderate Risk</b> and <b>Low Risk</b>
        are based on historical approval patterns within each cluster.
        </div>
        """,
        unsafe_allow_html=True
    )

# =======================================================
# Model Performance
# =======================================================
with tab_performance:
    st.subheader("Model Performance")

    table = pd.DataFrame([
        {
            "Model": "Baseline",
            "Threshold": metrics["baseline_threshold"],
            "Accuracy": metrics["baseline"]["accuracy"],
            "Precision": metrics["baseline"]["precision"],
            "Recall": metrics["baseline"]["recall"],
            "F1": metrics["baseline"]["f1"],
            "ROC-AUC": metrics["baseline"]["roc_auc"]
        },
        {
            "Model": "Improved",
            "Threshold": metrics["improved_threshold"],
            "Accuracy": metrics["improved"]["accuracy"],
            "Precision": metrics["improved"]["precision"],
            "Recall": metrics["improved"]["recall"],
            "F1": metrics["improved"]["f1"],
            "ROC-AUC": metrics["improved"]["roc_auc"]
        }
    ])

    display_table = table.copy()

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]:
        display_table[col] = (
            display_table[col]
            * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="soft-card">
        <b>Model improvement</b><br><br>
        The baseline model uses the standard 0.50 probability threshold.
        The improved model uses a threshold selected using validation data
        to improve the F1 score.
        </div>
        """,
        unsafe_allow_html=True
    )

    explained = metadata[
        "pca_explained_variance"
    ]

    st.metric(
        "PCA PC1 + PC2 Explained Variance",
        f"{sum(explained)*100:.1f}%"
    )

st.markdown("---")
st.caption(
    "CMP7005 • Retail Bank Loan Assessment & Customer Segmentation • Streamlit Demonstration"
)
