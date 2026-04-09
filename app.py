import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", layout="wide")

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- GOOGLE DRIVE LINKS ---------------- #
APPLICATION_URL = "https://drive.google.com/uc?id=1NnkxG5dp4c_BGH_CBdYZzFGNjVtsF2BQ"

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("application_record.csv")
    except:
        df = pd.read_csv(APPLICATION_URL)

    if 'DAYS_BIRTH' in df.columns:
        df['AGE'] = (-df['DAYS_BIRTH']) // 365

    if 'DAYS_EMPLOYED' in df.columns:
        df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED']) // 365

    return df

app_df = load_data()

# ---------------- INR FORMAT ---------------- #
def format_inr(num):
    if num >= 10000000:
        return f"₹{num/10000000:.2f} Cr"
    elif num >= 100000:
        return f"₹{num/100000:.2f} L"
    else:
        return f"₹{num:,.0f}"

# ---------------- SAFE ENCODE ---------------- #
def safe_encode(col, value):
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    return le.transform([le.classes_[0]])[0]

# ---------------- FEATURE IMPORTANCE ---------------- #
def get_feature_importance(model, columns):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    return None

# ---------------- SELECT BOX ---------------- #
def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI")
st.subheader("AI-Based Credit Risk & Approval System")

col1, col2, col3 = st.columns(3)

with col1:
    gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
    income = st.number_input("Annual Income (₹)", min_value=0)

with col2:
    income_type = select_box("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
    education = select_box("Education", encoders['NAME_EDUCATION_TYPE'].classes_)

with col3:
    family_status = select_box("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
    occupation = select_box("Occupation", encoders['OCCUPATION_TYPE'].classes_)

family_members = st.slider("Family Members", 1, 10, 2)
age = st.slider("Age", 18, 70, 30)
employment_years = st.slider("Employment Years", 0, 40, 5)

# ⭐ NEW: CREDIT SCORE
credit_score = st.slider("Credit Score", 300, 900, 650)

# ---------------- VALIDATION ---------------- #
errors = []
if gender == "Select": errors.append("Gender")
if income <= 0: errors.append("Income")
if income_type == "Select": errors.append("Income Type")
if education == "Select": errors.append("Education")
if family_status == "Select": errors.append("Family Status")
if occupation == "Select": errors.append("Occupation")

if errors:
    st.warning("Please fill: " + ", ".join(errors))

# ---------------- ANALYZE ---------------- #
if st.button("Analyze", disabled=len(errors) > 0):

    # Convert credit score (same logic as training)
    credit_score_model = (900 - credit_score) / 100

    input_df = pd.DataFrame([[
        safe_encode('CODE_GENDER', gender),
        income,
        safe_encode('NAME_INCOME_TYPE', income_type),
        safe_encode('NAME_EDUCATION_TYPE', education),
        safe_encode('NAME_FAMILY_STATUS', family_status),
        safe_encode('OCCUPATION_TYPE', occupation),
        family_members,
        age,
        employment_years,
        credit_score_model
    ]], columns=[
        'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
        'OCCUPATION_TYPE','CNT_FAM_MEMBERS','AGE','EMPLOYMENT_YEARS','CREDIT_SCORE'
    ])

    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]
    threshold = 0.65
    decision = "Approved" if prob > threshold else "Rejected"

    # ---------------- RESULT ---------------- #
    st.markdown("## Result")

    if decision == "Approved":
        st.success("Approved")
    else:
        st.error("Rejected")

    st.write(f"Confidence: {prob*100:.2f}%")
    st.progress(int(prob * 100))

    # ---------------- EXPLANATION ---------------- #
    st.markdown("## Explanation")

    if credit_score < 600:
        st.write("• Low credit score increases risk")

    if income < app_df['AMT_INCOME_TOTAL'].mean():
        st.write("• Income below average")

    if employment_years < app_df['EMPLOYMENT_YEARS'].mean():
        st.write("• Low employment stability")

    if family_members > app_df['CNT_FAM_MEMBERS'].mean():
        st.write("• High dependency")

    # ---------------- FEATURE IMPORTANCE ---------------- #
    st.markdown("## Feature Importance")

    feature_names = [
        'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
        'OCCUPATION_TYPE','CNT_FAM_MEMBERS','AGE','EMPLOYMENT_YEARS','CREDIT_SCORE'
    ]

    fi = get_feature_importance(model, feature_names)

    if fi is not None:
        st.bar_chart(fi.head(8))

    # ================== EDA ==================
    st.markdown("## 📊 Data Analysis")

    st.dataframe(app_df[['AMT_INCOME_TOTAL','AGE','EMPLOYMENT_YEARS','CNT_FAM_MEMBERS']].describe())
