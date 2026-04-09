import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI", layout="wide")

# ---------------- LOAD DATASETS ---------------- #
app_df = pd.read_csv("https://drive.google.com/uc?id=1NnkxG5dp4c_BGH_CBdYZzFGNjVtsF2BQ")
credit_df = pd.read_csv("https://drive.google.com/uc?id=188CysjlnrPZcxA1YuiYJH64HU5xXgHiD")

# ---------------- LOAD MODEL ---------------- #
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

# ---------------- SCORING SYSTEM ---------------- #
def calculate_score(income, credit_score, employment_years, family_members, age):
    score = 0
    reasons = []
    if income < 300000:
        return 0, ["Income below ₹3L (auto reject)"]
    if income > 1000000: score += 30; reasons.append("High income")
    elif income > 500000: score += 20; reasons.append("Moderate income")
    else: score += 10; reasons.append("Low income")
    if credit_score > 750: score += 30; reasons.append("Excellent credit score")
    elif credit_score > 650: score += 20; reasons.append("Good credit score")
    else: score += 10; reasons.append("Average credit score")
    if employment_years > 5: score += 15; reasons.append("Stable job")
    elif employment_years > 2: score += 10; reasons.append("Moderate job stability")
    if family_members <= 3: score += 15; reasons.append("Low financial burden")
    else: score += 5; reasons.append("High dependency")
    if 25 <= age <= 55: score += 10; reasons.append("Ideal working age")
    return score, reasons

# ---------------- UI ---------------- #
st.title("💳 CreditCheck AI – Full EDA and Scoring")
tab1, tab2 = st.tabs(["🧍 Individual", "📂 Bulk Upload"])

# ===== INDIVIDUAL ANALYSIS =====
with tab1:
    st.subheader("🔍 Individual Profile Input")
    
    col1, col2 = st.columns(2)
    gender = select_box("Gender", encoders['CODE_GENDER'].classes_)
    income = col1.number_input("Annual Income (₹)", min_value=0)
    age = col1.slider("Age", 18, 70, 30)
    credit_score = col2.slider("Credit Score", 300, 900, 650)
    employment_years = col2.slider("Employment Years", 0, 40, 5)
    family_members = st.slider("Family Members", 1, 10, 2)
    
    if st.button("Analyze Individual"):
        st.markdown("### 📊 Comparison with Dataset")
        
        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Your Income", f"₹{income:,}")
        col2.metric("Dataset Avg Income", f"₹{int(app_df['AMT_INCOME_TOTAL'].mean()):,}")
        
        col1, col2 = st.columns(2)
        col1.metric("Your Age", age)
        col2.metric("Dataset Avg Age", int(app_df['AGE'].mean()))
        
        # Histogram – Income
        fig1, ax1 = plt.subplots()
        ax1.hist(app_df['AMT_INCOME_TOTAL'], bins=30, alpha=0.6)
        ax1.axvline(income, color='red', label="You")
        ax1.set_title("Income Distribution")
        ax1.legend()
        st.pyplot(fig1)
        
        # Histogram – Age
        fig2, ax2 = plt.subplots()
        ax2.hist(app_df['AGE'], bins=30, alpha=0.6)
        ax2.axvline(age, color='red', label="You")
        ax2.set_title("Age Distribution")
        ax2.legend()
        st.pyplot(fig2)
        
        # Scatter – Employment vs Income
        fig3, ax3 = plt.subplots()
        ax3.scatter(app_df['EMPLOYMENT_YEARS'], app_df['AMT_INCOME_TOTAL'], alpha=0.6)
        ax3.scatter(employment_years, income, color='red', label="You")
        ax3.set_xlabel("Employment Years")
        ax3.set_ylabel("Income")
        ax3.set_title("Employment vs Income")
        ax3.legend()
        st.pyplot(fig3)
        
        # Score + Reasons
        score, reasons = calculate_score(income, credit_score, employment_years, family_members, age)
        st.write(f"### 🧠 Score: {score}/100")
        st.write("**Reasons:**")
        for r in reasons:
            st.write("✔", r)

# ===== BULK UPLOAD ANALYSIS =====
with tab2:
    st.subheader("📥 Upload Bulk CSV")
    uploaded = st.file_uploader("Select CSV", type="csv")
    
    if uploaded:
        df_original = pd.read_csv(uploaded)
        st.dataframe(df_original.head())
        
        df = encode_dataframe(df_original.copy())
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
            inc = df_original.loc[i,'AMT_INCOME_TOTAL']
            cs = df_original.loc[i,'CREDIT_SCORE']
            emp = df_original.loc[i,'EMPLOYMENT_YEARS']
            fam = df_original.loc[i,'CNT_FAM_MEMBERS']
            age_val = df_original.loc[i,'AGE']
            
            score, reasons = calculate_score(inc, cs, emp, fam, age_val)
            if score >= 70:
                decisions.append("Approved")
            elif score >= 50:
                decisions.append("Borderline")
            else:
                decisions.append("Rejected")
            confidences.append(round(probs[i]*100,2))
            scores.append(score)
            reasons_list.append(", ".join(reasons))
        
        df_out = df_original.copy()
        df_out["Decision"] = decisions
        df_out["Confidence (%)"] = confidences
        df_out["Score"] = scores
        df_out["Reasons"] = reasons_list
        
        st.write("### ✅ Bulk Results")
        st.dataframe(df_out)
        
        st.download_button("📥 Download Results", df_out.to_csv(index=False), "results.csv")
        
        # ===== Bulk EDA =====
        st.markdown("## 📊 Bulk Data Insights")
        
        # Decision distribution
        st.subheader("Decision Distribution")
        st.bar_chart(df_out['Decision'].value_counts())
        
        # Income vs Score scatter
        fig4, ax4 = plt.subplots()
        for decision in df_out['Decision'].unique():
            subset = df_out[df_out['Decision']==decision]
            ax4.scatter(subset['AMT_INCOME_TOTAL'], subset['Score'], label=decision, alpha=0.6)
        ax4.set_xlabel("Income")
        ax4.set_ylabel("Score")
        ax4.set_title("Income vs Score")
        ax4.legend()
        st.pyplot(fig4)
        
        # Confidence distribution
        fig5, ax5 = plt.subplots()
        ax5.hist(df_out['Confidence (%)'], bins=20, alpha=0.7)
        ax5.set_xlabel("Confidence %")
        ax5.set_title("Confidence % Distribution")
        st.pyplot(fig5)
        
        # Family impact
        fig6, ax6 = plt.subplots()
        for decision in df_out['Decision'].unique():
            subset = df_out[df_out['Decision']==decision]
            ax6.scatter(subset['CNT_FAM_MEMBERS'], subset['AMT_INCOME_TOTAL'], label=decision, alpha=0.6)
        ax6.set_xlabel("Family Members")
        ax6.set_ylabel("Income")
        ax6.set_title("Family Members Impact")
        ax6.legend()
        st.pyplot(fig6)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df_out.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig7, ax7 = plt.subplots(figsize=(8,6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax7)
            st.pyplot(fig7)
