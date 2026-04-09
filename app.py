import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", layout="wide")

# ---------------- LOAD FILES ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# ---------------- DATASETS ---------------- #
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

    # ❌ HARD RULE
    if income < 300000:
        return 0, ["Income below ₹3L (auto reject)"]

    # Income
    if income > 1000000:
        score += 30
        reasons.append("High income")
    elif income > 500000:
        score += 20
        reasons.append("Moderate income")
    else:
        score += 10
        reasons.append("Low income")

    # Credit Score
    if credit_score > 750:
        score += 30
        reasons.append("Excellent credit score")
    elif credit_score > 650:
        score += 20
        reasons.append("Good credit score")
    else:
        score += 10
        reasons.append("Average credit score")

    # Employment
    if employment_years > 5:
        score += 15
        reasons.append("Stable job")
    elif employment_years > 2:
        score += 10
        reasons.append("Moderate job stability")

    # Family
    if family_members <= 3:
        score += 15
        reasons.append("Low financial burden")
    else:
        score += 5
        reasons.append("High dependency")

    # Age
    if 25 <= age <= 55:
        score += 10
        reasons.append("Ideal working age")

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
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        # ⭐ SCORING
        score, reasons = calculate_score(income, credit_score, employment_years, family_members, age)

        # ⭐ FINAL DECISION
        if income < 300000:
            decision = "Rejected"
            final_conf = prob * 0.3
        elif score >= 70:
            decision = "Approved"
            final_conf = prob
        elif score >= 50:
            decision = "Borderline"
            final_conf = prob * 0.7
        else:
            decision = "Rejected"
            final_conf = prob * 0.5

        # DISPLAY
        st.markdown("## 🎯 Result")
        if decision == "Approved": st.success("Approved")
        elif decision == "Rejected": st.error("Rejected")
        else: st.warning("Borderline")
        st.write(f"Confidence: {final_conf*100:.2f}%")

        st.markdown("### 🧠 Score Breakdown")
        st.progress(score)
        st.write(f"Score: {score}/100")
        for r in reasons:
            st.write(f"✔ {r}")

        # ----------------- INDIVIDUAL INSIGHTS -----------------
        st.markdown("## 📊 Your Profile vs Dataset")

        # Compare histogram Income
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        for dec in ['Approved','Borderline','Rejected']:
            subset = app_df.copy()
            subset_score, _ = calculate_score(subset['AMT_INCOME_TOTAL'], subset.get('CREDIT_SCORE',650),
                                              subset['EMPLOYMENT_YEARS'], subset['CNT_FAM_MEMBERS'], subset['AGE'])
            # simulate decisions
            subset_dec = ['Approved' if s>=70 else 'Borderline' if s>=50 else 'Rejected' for s in subset_score]
            ax.hist(subset['AMT_INCOME_TOTAL'][np.array(subset_dec)==dec], bins=30, alpha=0.5, label=dec)
        ax.axvline(income, color='red', linewidth=2, label='You')
        ax.set_xlabel("Income"); ax.set_ylabel("Count"); ax.legend()
        st.pyplot(fig)

        # Age distribution
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        for dec in ['Approved','Borderline','Rejected']:
            subset = app_df.copy()
            subset_score, _ = calculate_score(subset['AMT_INCOME_TOTAL'], subset.get('CREDIT_SCORE',650),
                                              subset['EMPLOYMENT_YEARS'], subset['CNT_FAM_MEMBERS'], subset['AGE'])
            subset_dec = ['Approved' if s>=70 else 'Borderline' if s>=50 else 'Rejected' for s in subset_score]
            ax.hist(subset['AGE'][np.array(subset_dec)==dec], bins=30, alpha=0.5, label=dec)
        ax.axvline(age, color='red', linewidth=2, label='You')
        ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
        st.pyplot(fig)

        # Employment vs Income
        st.subheader("Employment Years vs Income")
        fig, ax = plt.subplots()
        for dec in ['Approved','Borderline','Rejected']:
            subset = app_df.copy()
            subset_score, _ = calculate_score(subset['AMT_INCOME_TOTAL'], subset.get('CREDIT_SCORE',650),
                                              subset['EMPLOYMENT_YEARS'], subset['CNT_FAM_MEMBERS'], subset['AGE'])
            subset_dec = ['Approved' if s>=70 else 'Borderline' if s>=50 else 'Rejected' for s in subset_score]
            scatter_data = subset[np.array(subset_dec)==dec]
            ax.scatter(scatter_data['EMPLOYMENT_YEARS'], scatter_data['AMT_INCOME_TOTAL'], alpha=0.6, label=dec)
        ax.scatter(employment_years, income, color='red', s=100, label='You')
        ax.set_xlabel("Employment Years"); ax.set_ylabel("Income"); ax.legend()
        st.pyplot(fig)

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
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype(str).str.title()
        original_credit_score = df['CREDIT_SCORE'].copy()
        df = encode_dataframe(df)
        if 'CREDIT_SCORE' in df.columns:
            df['CREDIT_SCORE'] = (900 - df['CREDIT_SCORE']) / 100
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        df_scaled = scaler.transform(df)
        probs = model.predict_proba(df_scaled)[:,1]

        decisions, confidences, scores, reasons_list = [], [], [], []
        for i in range(len(df)):
            income_i = df.iloc[i]['AMT_INCOME_TOTAL']
            credit_score_raw = original_credit_score.iloc[i]
            emp = df.iloc[i]['EMPLOYMENT_YEARS']
            fam = df.iloc[i]['CNT_FAM_MEMBERS']
            age_i = df.iloc[i]['AGE']
            prob = probs[i]
            score_i, reasons = calculate_score(income_i, credit_score_raw, emp, fam, age_i)
            if income_i < 300000:
                decision = "Rejected"
                conf = prob*0.3
            elif score_i>=70:
                decision = "Approved"
                conf = prob
            elif score_i>=50:
                decision = "Borderline"
                conf = prob*0.7
            else:
                decision = "Rejected"
                conf = prob*0.5
            decisions.append(decision)
            confidences.append(round(conf*100,2))
            scores.append(score_i)
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
        st.markdown("## 📊 Bulk Dataset Insights")

        st.subheader("Decision Count")
        fig, ax = plt.subplots()
        sns.countplot(x='Decision', data=result_df, palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("Income Distribution by Decision")
        fig, ax = plt.subplots()
        sns.boxplot(x='Decision', y='AMT_INCOME_TOTAL', data=result_df, palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("Credit Score Distribution by Decision")
        fig, ax = plt.subplots()
        sns.boxplot(x='Decision', y='CREDIT_SCORE', data=result_df, palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("Employment vs Income")
        fig, ax = plt.subplots()
        sns.scatterplot(x='EMPLOYMENT_YEARS', y='AMT_INCOME_TOTAL', hue='Decision', data=result_df, palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("Score vs Income")
        fig, ax = plt.subplots()
        sns.scatterplot(x='AMT_INCOME_TOTAL', y='Score', hue='Decision', data=result_df, palette="Set2", ax=ax)
        st.pyplot(fig)
