import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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

def preprocess_input(df):
    df = df.copy()

    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].astype(str)

            df[col] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else le.transform([le.classes_[0]])[0]
            )
    return df

def transform_credit(score):
    # same logic as training
    return (900 - score) / 100

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI")

tab1, tab2 = st.tabs(["🧍 Individual", "📂 Bulk Upload"])

# =====================================================
# 🧍 INDIVIDUAL
# =====================================================
with tab1:

    gender = st.selectbox("Gender", encoders['CODE_GENDER'].classes_)
    income = st.number_input("Income", min_value=0)
    income_type = st.selectbox("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
    education = st.selectbox("Education", encoders['NAME_EDUCATION_TYPE'].classes_)
    family_status = st.selectbox("Family Status", encoders['NAME_FAMILY_STATUS'].classes_)
    occupation = st.selectbox("Occupation", encoders['OCCUPATION_TYPE'].classes_)

    family_members = st.slider("Family Members", 1, 10, 2)
    age = st.slider("Age", 18, 70, 30)
    employment_years = st.slider("Employment Years", 0, 40, 5)
    credit_score = st.slider("Credit Score", 300, 900, 650)

    if st.button("Predict"):

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
            'CREDIT_SCORE': transform_credit(credit_score)
        }

        df = pd.DataFrame([input_dict])

        # Align features
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]

        df_scaled = scaler.transform(df)

        prob = model.predict_proba(df_scaled)[0][1]

        st.success("Approved" if prob > 0.65 else "Rejected")
        st.write(f"Confidence: {prob*100:.2f}%")

# =====================================================
# 📂 BULK UPLOAD (FIXED)
# =====================================================
with tab2:

    st.subheader("Upload CSV")

    sample = pd.DataFrame(columns=feature_names)
    st.download_button("Download Sample CSV", sample.to_csv(index=False), "sample.csv")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview")
        st.dataframe(df.head())

        try:
            # ---------------- CLEAN ---------------- #
            for col in df.select_dtypes(include='object'):
                df[col] = df[col].astype(str).str.strip()

            # ---------------- CREDIT SCORE ---------------- #
            if "CREDIT_SCORE" in df.columns:
                df['CREDIT_SCORE'] = df['CREDIT_SCORE'].apply(transform_credit)

            # ---------------- ENCODE ---------------- #
            df = preprocess_input(df)

            # ---------------- FEATURE ALIGN ---------------- #
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_names]

            # ---------------- SCALE ---------------- #
            df_scaled = scaler.transform(df)

            probs = model.predict_proba(df_scaled)[:,1] * 100

            df['Decision'] = np.where(probs > 65, "Approved",
                              np.where(probs < 45, "Rejected", "Borderline"))

            df['Confidence'] = probs

            # ---------------- SUMMARY ---------------- #
            st.markdown("### 📊 Summary")
            st.metric("Total", len(df))
            st.metric("Approved", (df['Decision']=="Approved").sum())
            st.metric("Rejected", (df['Decision']=="Rejected").sum())

            st.dataframe(df)

            # ---------------- EDA ---------------- #
            st.markdown("## 📊 Data Insights")

            # Correlation
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            cax = ax.matshow(corr)
            fig.colorbar(cax)
            st.pyplot(fig)

            # Income distribution
            if "AMT_INCOME_TOTAL" in df.columns:
                fig, ax = plt.subplots()
                ax.hist(df['AMT_INCOME_TOTAL'], bins=20)
                st.pyplot(fig)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                st.markdown("### Feature Importance")
                imp = pd.Series(model.feature_importances_, index=feature_names)
                st.bar_chart(imp.sort_values(ascending=False))

        except Exception as e:
            st.error(f"Error: {e}")
