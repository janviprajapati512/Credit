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
feature_importance = pd.read_csv("feature_importance.csv", index_col=0)

# ---------------- GOOGLE DRIVE LINKS ---------------- #
APPLICATION_URL = "https://drive.google.com/uc?id=1NnkxG5dp4c_BGH_CBdYZzFGNjVtsF2BQ"
CREDIT_URL = "https://drive.google.com/uc?id=188CysjlnrPZcxA1YuiYJH64HU5xXgHiD"

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    try:
        app = pd.read_csv("application_record.csv")
        credit = pd.read_csv("credit_record.csv")
    except:
        app = pd.read_csv(APPLICATION_URL)
        credit = pd.read_csv(CREDIT_URL)

    # Feature engineering
    if 'DAYS_BIRTH' in app.columns:
        app['AGE'] = (-app['DAYS_BIRTH']) // 365

    if 'DAYS_EMPLOYED' in app.columns:
        app['EMPLOYMENT_YEARS'] = (-app['DAYS_EMPLOYED']) // 365

    return app, credit

app_df, credit_df = load_data()

# ---------------- FORMAT INR ---------------- #
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

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI")
st.subheader("AI-Based Credit Risk & Approval System")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", encoders['CODE_GENDER'].classes_)
    income = st.number_input("Annual Income (₹)", min_value=0)

with col2:
    income_type = st.selectbox("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
    education = st.selectbox("Education", encoders['NAME_EDUCATION_TYPE'].classes_)

with col3:
    family_status = st.selectbox("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
    occupation = st.selectbox("Occupation", encoders['OCCUPATION_TYPE'].classes_)

family_members = st.slider("Family Members", 1, 10, 2)
age = st.slider("Age", 18, 70, 30)
employment_years = st.slider("Employment Years", 0, 40, 5)

valid = income > 0

# ---------------- ANALYZE ---------------- #
if st.button("Analyze", disabled=not valid):

    input_df = pd.DataFrame([[
        safe_encode('CODE_GENDER', gender),
        income,
        safe_encode('NAME_INCOME_TYPE', income_type),
        safe_encode('NAME_EDUCATION_TYPE', education),
        safe_encode('NAME_FAMILY_STATUS', family_status),
        safe_encode('OCCUPATION_TYPE', occupation),
        family_members,
        age,
        employment_years
    ]], columns=[
        'CODE_GENDER','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
        'OCCUPATION_TYPE','CNT_FAM_MEMBERS','AGE','EMPLOYMENT_YEARS'
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

    if income < app_df['AMT_INCOME_TOTAL'].mean():
        st.write("• Income below average")

    if employment_years < app_df['EMPLOYMENT_YEARS'].mean():
        st.write("• Low employment stability")

    if family_members > app_df['CNT_FAM_MEMBERS'].mean():
        st.write("• High dependency")

    # ---------------- FEATURE IMPORTANCE ---------------- #
    st.markdown("## Feature Importance")
    st.bar_chart(feature_importance.head(8))

    # ================== EDA ==================
    st.markdown("## 📊 Data Analysis")

    st.subheader("Statistical Summary")
    st.dataframe(app_df[['AMT_INCOME_TOTAL','AGE','EMPLOYMENT_YEARS','CNT_FAM_MEMBERS']].describe())

    colA, colB = st.columns(2)

    with colA:
        st.write("Income Distribution")
        st.bar_chart(app_df['AMT_INCOME_TOTAL'].value_counts().head(50))

    with colB:
        st.write("Age Distribution")
        st.bar_chart(app_df['AGE'].value_counts())

    st.subheader("Correlation Heatmap")
    corr = app_df.select_dtypes(include=np.number).corr()
    st.dataframe(corr.style.background_gradient(cmap='Blues'))

    # ---------------- COMPARISON ---------------- #
    st.subheader("Your Profile vs Dataset")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Your Income", format_inr(income))
        st.metric("Avg Income", format_inr(int(app_df['AMT_INCOME_TOTAL'].mean())))

    with c2:
        st.metric("Your Age", age)
        st.metric("Avg Age", int(app_df['AGE'].mean()))

    compare = pd.DataFrame({
        "Metric": ["Income","Age","Employment"],
        "You": [income, age, employment_years],
        "Average": [
            app_df['AMT_INCOME_TOTAL'].mean(),
            app_df['AGE'].mean(),
            app_df['EMPLOYMENT_YEARS'].mean()
        ]
    })

    st.bar_chart(compare.set_index("Metric"))
