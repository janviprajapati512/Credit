import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", layout="wide")
sns.set_style("whitegrid")

# ---------------- LOAD FILES ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# Load dataset for EDA
app_df = pd.read_csv("https://drive.google.com/uc?id=1NnkxG5dp4c_BGH_CBdYZzFGNjVtsF2BQ")
credit_df = pd.read_csv("https://drive.google.com/uc?id=188CysjlnrPZcxA1YuiYJH64HU5xXgHiD")

# ---------------- HELPERS ---------------- #
def safe_encode(col, value):
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    return le.transform([le.classes_[0]])[0]

def encode_dataframe(df):
    for col in encoders:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: safe_encode(col, x))
    return df

def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

# ---------------- SCORING SYSTEM ---------------- #
def calculate_score(income, credit_score, employment_years, family_members, age):
    score = 0
    reasons = []

    if income < 300000:
        return 0, ["Income below ₹3L (auto reject)"]

    if income > 1000000:
        score += 30; reasons.append("High income")
    elif income > 500000:
        score += 20; reasons.append("Moderate income")
    else:
        score += 10; reasons.append("Low income")

    if credit_score > 750:
        score += 30; reasons.append("Excellent credit score")
    elif credit_score > 650:
        score += 20; reasons.append("Good credit score")
    else:
        score += 10; reasons.append("Average credit score")

    if employment_years > 5:
        score += 15; reasons.append("Stable job")
    elif employment_years > 2:
        score += 10; reasons.append("Moderate job stability")

    if family_members <= 3:
        score += 15; reasons.append("Low financial burden")
    else:
        score += 5; reasons.append("High dependency")

    if 25 <= age <= 55:
        score += 10; reasons.append("Ideal working age")

    return score, reasons

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI")
st.subheader("AI + Rule-Based Credit Approval System")

tab1, tab2 = st.tabs(["Individual", "Bulk Upload"])

# =====================================================
# 🧍 INDIVIDUAL
# =====================================================
with tab1:
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
    credit_score = st.slider("Credit Score", 300, 900, 650)

    errors = []
    if gender == "Select": errors.append("Gender")
    if income <= 0: errors.append("Income")
    if income_type == "Select": errors.append("Income Type")
    if education == "Select": errors.append("Education")
    if family_status == "Select": errors.append("Family Status")
    if occupation == "Select": errors.append("Occupation")

    if errors:
        st.warning("Fill all fields: " + ", ".join(errors))

    if st.button("Analyze", disabled=len(errors) > 0):
        credit_score_model = (900 - credit_score) / 100
        input_dict = {
            'CODE_GENDER': safe_encode('CODE_GENDER', gender),
            'AMT_INCOME_TOTAL': income,
            'NAME_INCOME_TYPE': safe_encode('NAME_INCOME_TYPE', income_type),
            'NAME_EDUCATION_TYPE': safe_encode('NAME_EDUCATION_TYPE', education),
            'NAME_FAMILY_STATUS': safe_encode('NAME_FAMILY_STATUS', family_status),
            'OCCUPATION_TYPE': safe_encode('OCCUPATION_TYPE', occupation),
            'CNT_FAM_MEMBERS': family_members,
            'AGE': age,
            'EMPLOYMENT_YEARS': employment_years,
            'CREDIT_SCORE': credit_score_model
        }
        input_df = pd.DataFrame([input_dict])
        for col in feature_names:
            if col not in input_df.columns: input_df[col] = 0
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]
        score, reasons = calculate_score(income, credit_score, employment_years, family_members, age)

        if income < 300000:
            decision = "Rejected"; final_conf = prob * 0.3
        elif score >= 70:
            decision = "Approved"; final_conf = prob
        elif score >= 50:
            decision = "Borderline"; final_conf = prob * 0.7
        else:
            decision = "Rejected"; final_conf = prob * 0.5

        st.markdown("## 🎯 Result")
        if decision == "Approved": st.success("Approved")
        elif decision == "Rejected": st.error("Rejected")
        else: st.warning("Borderline")
        st.write(f"Confidence: {final_conf*100:.2f}%")

        st.markdown("### 🧠 Score Breakdown")
        st.progress(score)
        st.write(f"Score: {score}/100")
        for r in reasons: st.write(f"✔ {r}")

        # --------- Individual EDA ---------
        st.markdown("## 📊 Your Profile Analysis")
        col1, col2 = st.columns(2)
        col1.metric("Your Income", f"₹{income:,}")
        col2.metric("Average Income", f"₹{int(app_df['AMT_INCOME_TOTAL'].mean()):,}")
        col1, col2 = st.columns(2)
        col1.metric("Your Age", age)
        col2.metric("Average Age", int(app_df['AGE'].mean()))

        compare_df = pd.DataFrame({
            "Metric": ["Income", "Age", "Employment"],
            "You": [income, age, employment_years],
            "Average": [
                app_df['AMT_INCOME_TOTAL'].mean(),
                app_df['AGE'].mean(),
                app_df['EMPLOYMENT_YEARS'].mean()
            ]
        })
        st.bar_chart(compare_df.set_index("Metric"))

# =====================================================
# 📂 BULK UPLOAD
# =====================================================
with tab2:
    st.subheader("Upload CSV")
    sample = pd.DataFrame({
        'CODE_GENDER': ['M'],
        'AMT_INCOME_TOTAL': [500000],
        'NAME_INCOME_TYPE': ['Working'],
        'NAME_EDUCATION_TYPE': ['Higher education'],
        'NAME_FAMILY_STATUS': ['Married'],
        'OCCUPATION_TYPE': ['Managers'],
        'CNT_FAM_MEMBERS': [2],
        'AGE': [30],
        'EMPLOYMENT_YEARS': [5],
        'CREDIT_SCORE': [700]
    })
    st.download_button("📥 Download Sample CSV", sample.to_csv(index=False), "sample.csv")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df_original = pd.read_csv(file)
        df = df_original.copy()
        st.write("Preview")
        st.dataframe(df_original.head())

        try:
            # ---------------- CLEAN ---------------- #
            for col in df.select_dtypes(include='object'):
                df[col] = df[col].astype(str).str.title()
            original_credit_score = df['CREDIT_SCORE'].copy()
            df = encode_dataframe(df)
            if 'CREDIT_SCORE' in df.columns:
                df['CREDIT_SCORE'] = (900 - df['CREDIT_SCORE']) / 100
            for col in feature_names:
                if col not in df.columns: df[col] = 0
            df = df[feature_names]
            df_scaled = scaler.transform(df)
            probs = model.predict_proba(df_scaled)[:, 1]

            decisions, confidences, scores, reasons_list = [], [], [], []
            for i in range(len(df)):
                income = df.iloc[i]['AMT_INCOME_TOTAL']
                credit_score_raw = original_credit_score.iloc[i]
                emp = df.iloc[i]['EMPLOYMENT_YEARS']
                fam = df.iloc[i]['CNT_FAM_MEMBERS']
                age = df.iloc[i]['AGE']
                prob = probs[i]
                score, reasons = calculate_score(income, credit_score_raw, emp, fam, age)

                if income < 300000: decision = "Rejected"; conf = prob * 0.3
                elif score >= 70: decision = "Approved"; conf = prob
                elif score >= 50: decision = "Borderline"; conf = prob * 0.7
                else: decision = "Rejected"; conf = prob * 0.5

                decisions.append(decision)
                confidences.append(round(conf * 100, 2))
                scores.append(score)
                reasons_list.append(", ".join(reasons))

            result_df = df_original.copy()
            result_df['Decision'] = decisions
            result_df['Confidence (%)'] = confidences
            result_df['Score'] = scores
            result_df['Reasons'] = reasons_list
            st.success("Processed Successfully ✅")
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", data=csv, file_name="credit_results.csv", mime="text/csv")

            # ----------------- BULK EDA -----------------
            numeric_cols = ['AMT_INCOME_TOTAL', 'CREDIT_SCORE', 'EMPLOYMENT_YEARS', 'Score']
            for col in numeric_cols: result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df_clean = result_df.dropna(subset=['AMT_INCOME_TOTAL', 'CREDIT_SCORE', 'EMPLOYMENT_YEARS', 'Score', 'Decision'])
            result_df_clean['Decision'] = result_df_clean['Decision'].astype('category')

            st.subheader("Decision Count")
            fig, ax = plt.subplots()
            sns.countplot(x='Decision', data=result_df_clean, palette="Set2", ax=ax)
            st.pyplot(fig)

            st.subheader("Income Distribution by Decision")
            fig, ax = plt.subplots()
            sns.boxplot(x='Decision', y='AMT_INCOME_TOTAL', data=result_df_clean, palette="Set2", ax=ax)
            st.pyplot(fig)

            st.subheader("Credit Score Distribution by Decision")
            fig, ax = plt.subplots()
            sns.boxplot(x='Decision', y='CREDIT_SCORE', data=result_df_clean, palette="Set2", ax=ax)
            st.pyplot(fig)

            st.subheader("Employment vs Income")
            fig, ax = plt.subplots()
            sns.scatterplot(x='EMPLOYMENT_YEARS', y='AMT_INCOME_TOTAL', hue='Decision', data=result_df_clean, palette="Set2", ax=ax)
            st.pyplot(fig)

            st.subheader("Score vs Income")
            fig, ax = plt.subplots()
            sns.scatterplot(x='AMT_INCOME_TOTAL', y='Score', hue='Decision', data=result_df_clean, palette="Set2", ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
