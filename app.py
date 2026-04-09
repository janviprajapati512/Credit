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
feature_names = joblib.load("features.pkl")  # ⭐ CRITICAL FIX

# ---------------- DATA URL ---------------- #
APPLICATION_URL = "https://drive.google.com/uc?id=1NnkxG5dp4c_BGH_CBdYZzFGNjVtsF2BQ"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("application_record.csv")
    except:
        df = pd.read_csv(APPLICATION_URL)

    df['AGE'] = (-df['DAYS_BIRTH']) // 365
    df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED']) // 365

    return df

app_df = load_data()

# ---------------- HELPERS ---------------- #
def safe_encode(col, value):
    le = encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    return le.transform([le.classes_[0]])[0]

def select_box(label, options):
    return st.selectbox(label, ["Select"] + list(options))

def format_inr(num):
    if num >= 10000000:
        return f"₹{num/10000000:.2f} Cr"
    elif num >= 100000:
        return f"₹{num/100000:.2f} L"
    return f"₹{num:,.0f}"

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI")
st.subheader("AI-Based Credit Approval System")

tab1, tab2 = st.tabs(["🧍 Individual", "📂 Bulk Upload"])

# =====================================================
# 🧍 TAB 1 - INDIVIDUAL
# =====================================================
with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
        income = st.number_input("Income (₹)", min_value=0)

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

    # Validation
    errors = []
    if gender == "Select": errors.append("Gender")
    if income <= 0: errors.append("Income")
    if income_type == "Select": errors.append("Income Type")
    if education == "Select": errors.append("Education")
    if family_status == "Select": errors.append("Family Status")
    if occupation == "Select": errors.append("Occupation")

    if errors:
        st.warning("Fill all fields: " + ", ".join(errors))

    # ---------------- PREDICT ---------------- #
    if st.button("Analyze", disabled=len(errors) > 0):

        credit_score_model = (900 - credit_score) / 100

        # ⭐ CREATE FULL FEATURE DICT
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

        # ⭐ MATCH TRAINING FEATURES EXACTLY
        input_df = pd.DataFrame([input_dict])[feature_names]

        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        decision = "Approved" if prob > 0.65 else "Rejected"

        # RESULT
        st.markdown("## Result")
        st.success(decision) if decision == "Approved" else st.error(decision)
        st.progress(int(prob * 100))
        st.write(f"Confidence: {prob*100:.2f}%")

        # EXPLANATION
        st.markdown("## Explanation")

        if credit_score < 600:
            st.write("• Low credit score")
        if income < app_df['AMT_INCOME_TOTAL'].mean():
            st.write("• Income below average")
        if employment_years < app_df['EMPLOYMENT_YEARS'].mean():
            st.write("• Low job stability")

        # FEATURE IMPORTANCE
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feature_names)
            st.bar_chart(fi.sort_values(ascending=False).head(8))

        # EDA
        st.markdown("## 📊 EDA")
        st.dataframe(app_df[['AMT_INCOME_TOTAL','AGE','EMPLOYMENT_YEARS']].describe())
        st.bar_chart(app_df['AMT_INCOME_TOTAL'].head(50))


# =====================================================
# 📂 TAB 2 - BULK
# =====================================================
with tab2:

    st.subheader("Upload CSV for Bulk Prediction")

    sample = pd.DataFrame(columns=feature_names)
    st.download_button("Download Sample CSV", sample.to_csv(index=False), "sample.csv")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        try:
            df = df[feature_names]
            df_scaled = scaler.transform(df)

            probs = model.predict_proba(df_scaled)[:,1]
            df['Prediction'] = np.where(probs > 0.65, "Approved", "Rejected")
            df['Confidence'] = probs

            st.success("Processed Successfully")
            st.dataframe(df)

            st.bar_chart(df['Prediction'].value_counts())

        except Exception as e:
            st.error(f"Error: {e}")
