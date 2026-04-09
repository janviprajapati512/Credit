import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from difflib import get_close_matches

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI Pro", page_icon="💳", layout="wide")

# ---------------- UI ---------------- #
st.markdown("""
<style>
body {background:#0e1117; color:white;}
.title {font-size:36px;font-weight:bold;color:#4da6ff;text-align:center;}
.subtitle {text-align:center;color:#aaa;margin-bottom:25px;}
.stButton>button {background:#4da6ff;color:black;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">CreditCheck AI PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable • Bias-Reduced • Enterprise Grade</div>', unsafe_allow_html=True)

# ---------------- LOAD ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- HELPERS ---------------- #
def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

def safe_encode(col, value):
    classes = list(encoders[col].classes_)
    value = str(value).strip().title()

    if value in classes:
        return encoders[col].transform([value])[0]

    match = get_close_matches(value, classes, n=1, cutoff=0.6)
    if match:
        return encoders[col].transform([match[0]])[0]

    return -1

# 🔥 NORMALIZATION (reduces bias)
def normalize(df):
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'] / 100000
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'] / 10
    return df

# 🔥 BETTER CREDIT SCORE HANDLING (less dominance)
def transform_credit(score):
    return (900 - score) / 200   # previously /100 → now reduced impact

# 🔥 EXPLAINABILITY
def explain(df):
    reasons = []

    if df['AMT_INCOME_TOTAL'][0] > 0.5:
        reasons.append("High income increases repayment capacity")

    if df['CNT_FAM_MEMBERS'][0] > 0.5:
        reasons.append("Higher family size may increase financial burden")

    if df['CREDIT_SCORE'][0] < 2:
        reasons.append("Strong credit history improves approval chances")

    if df['CREDIT_SCORE'][0] > 4:
        reasons.append("Lower credit score indicates higher risk")

    return reasons

# ---------------- TABS ---------------- #
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📂 Bulk Analysis", "📊 EDA Dashboard"])

# =========================================================
# 🔮 PREDICTION
# =========================================================
with tab1:

    st.subheader("Applicant Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
        income = st.number_input("Income", min_value=0)

    with c2:
        income_type = select_box("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
        education = select_box("Education", encoders['NAME_EDUCATION_TYPE'].classes_)

    with c3:
        family_status = select_box("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
        occupation = select_box("Occupation", encoders['OCCUPATION_TYPE'].classes_)

    family_members = st.slider("Family Members", 1, 10, 2)
    credit_score = st.slider("Credit Score", 300, 900, 650)

    # Validation
    errors = []
    if gender == "Select": errors.append("Gender")
    if income <= 0: errors.append("Income")

    if errors:
        st.warning("Please fill: " + ", ".join(errors))

    if st.button("Analyze", disabled=len(errors) > 0):

        df = pd.DataFrame([[ 
            safe_encode('CODE_GENDER', gender),
            income,
            safe_encode('NAME_INCOME_TYPE', income_type),
            safe_encode('NAME_EDUCATION_TYPE', education),
            safe_encode('NAME_FAMILY_STATUS', family_status),
            safe_encode('OCCUPATION_TYPE', occupation),
            family_members,
            transform_credit(credit_score)
        ]], columns=[
            'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
            'OCCUPATION_TYPE','CNT_FAM_MEMBERS','CREDIT_SCORE'
        ])

        df = normalize(df)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        decision = "Approved" if pred == 1 else "Rejected"

        st.markdown("### Result")
        st.success(f"{decision} ({prob:.2f}%)")

        st.progress(int(prob))

        # Explainability
        st.markdown("### Why this decision?")
        for r in explain(df):
            st.write("•", r)

# =========================================================
# 📂 BULK
# =========================================================
with tab2:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview")
        st.dataframe(df.head())

        # Clean
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).str.strip().str.title()

        df['CREDIT_SCORE'] = df['CREDIT_SCORE'].apply(transform_credit)
        df = normalize(df)

        preds = model.predict(df)
        probs = model.predict_proba(df)[:,1] * 100

        df['Decision'] = ["Approved" if p==1 else "Rejected" for p in preds]
        df['Confidence (%)'] = np.round(probs,2)

        st.markdown("### Results")
        st.dataframe(df)

        st.download_button("Download Results", df.to_csv(index=False), "results.csv")

# =========================================================
# 📊 EDA DASHBOARD
# =========================================================
with tab3:

    st.subheader("Exploratory Data Analysis")

    file = st.file_uploader("Upload Dataset", type=["csv"], key="eda")

    if file:
        df = pd.read_csv(file)

        st.write("Dataset Preview")
        st.dataframe(df.head())

        # Stats
        st.markdown("### 📈 Statistical Summary")
        st.write(df.describe())

        # Missing values
        st.markdown("### ❗ Missing Values")
        st.write(df.isnull().sum())

        # Income Distribution
        st.markdown("### 💰 Income Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['AMT_INCOME_TOTAL'], bins=20)
        st.pyplot(fig)

        # Credit Score Distribution
        st.markdown("### 📊 Credit Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['CREDIT_SCORE'], bins=20)
        st.pyplot(fig)

        # Correlation
        st.markdown("### 🔗 Correlation Heatmap")
        corr = df.corr(numeric_only=True)

        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        st.pyplot(fig)

        # Insight
        st.markdown("### 🧠 Insights")
        st.info("Use this analysis to detect bias, feature importance, and data imbalance.")