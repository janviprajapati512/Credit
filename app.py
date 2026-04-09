import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", layout="wide")

# ---------------- LOAD FILES ---------------- #
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

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

# ⭐ BUSINESS RULE
def apply_business_rules(income, prob):
    if income < 300000:
        return "Rejected", prob, "Income below ₹3L (auto reject rule)"
    
    if prob > 0.65:
        return "Approved", prob, "Model approved"
    elif prob < 0.45:
        return "Rejected", prob, "Model rejected"
    else:
        return "Borderline", prob, "Manual review suggested"

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI")
st.subheader("AI + Rule-Based Credit Approval System")

tab1, tab2 = st.tabs(["🧍 Individual", "📂 Bulk Upload"])

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

        # Fix columns
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_names]

        # Scale
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]

        # ⭐ APPLY RULE
        decision, prob, reason = apply_business_rules(income, prob)

        st.markdown("## Result")

        if decision == "Approved":
            st.success("Approved")
        elif decision == "Rejected":
            st.error("Rejected")
        else:
            st.warning("Borderline")

        st.write(f"Confidence: {prob*100:.2f}%")
        st.write(f"Reason: {reason}")

# =====================================================
# 📂 BULK
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

    st.download_button("Download Sample CSV", sample.to_csv(index=False), "sample.csv")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview")
        st.dataframe(df.head())

        try:
            # Encode
            df = encode_dataframe(df)

            # Credit score transform
            if 'CREDIT_SCORE' in df.columns:
                df['CREDIT_SCORE'] = (900 - df['CREDIT_SCORE']) / 100

            # Fix columns
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_names]

            # Scale
            df_scaled = scaler.transform(df)

            probs = model.predict_proba(df_scaled)[:,1]

            decisions = []
            reasons = []

            for i in range(len(df)):
                income = df.iloc[i]['AMT_INCOME_TOTAL']
                d, p, r = apply_business_rules(income, probs[i])
                decisions.append(d)
                reasons.append(r)

            df['Decision'] = decisions
            df['Confidence'] = probs * 100
            df['Reason'] = reasons

            st.success("Processed Successfully")
            st.dataframe(df)

            st.bar_chart(df['Decision'].value_counts())

        except Exception as e:
            st.error(f"Error: {e}")
