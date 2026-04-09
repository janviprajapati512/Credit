import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="CreditCheck AI - Advanced EDA", layout="wide")
st.title("📊 CreditCheck AI - Advanced Exploratory Data Analysis")

# ---------------- LOAD DATA ---------------- #
st.sidebar.header("Select Dataset")
dataset_option = st.sidebar.selectbox("Dataset", ["App Dataset", "Credit Dataset"])

if dataset_option == "App Dataset":
    df = pd.read_csv("https://drive.google.com/uc?id=1NnkxG5dp4c_BGH_CBdYZzFGNjVtsF2BQ")
else:
    df = pd.read_csv("https://drive.google.com/uc?id=188CysjlnrPZcxA1YuiYJH64HU5xXgHiD")

df.columns = df.columns.str.strip().str.upper()

st.subheader(f"Dataset: {dataset_option}")
st.write("Preview of data:")
st.dataframe(df.head())

# ---------------- BASIC STATISTICS ---------------- #
st.subheader("🧮 Basic Statistics")
st.write("Shape of dataset:", df.shape)
st.write("Columns:", list(df.columns))
st.write("Missing values per column:")
st.dataframe(df.isnull().sum())
st.write("Data types:")
st.dataframe(df.dtypes)

st.write("Descriptive statistics for numerical columns:")
st.dataframe(df.describe())

# ---------------- INTERACTIVE FILTERS ---------------- #
st.sidebar.header("Filter Data")
filters = {}
for col in df.select_dtypes(include=['object']).columns:
    options = df[col].dropna().unique().tolist()
    selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
    filters[col] = selected
    df = df[df[col].isin(selected)]

st.subheader("Filtered Dataset Preview")
st.dataframe(df.head())

# ---------------- NUMERICAL DISTRIBUTIONS ---------------- #
st.subheader("📈 Numerical Features Distribution")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    st.write(f"**{col}**")
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, bins=30, color="skyblue", ax=ax)
    st.pyplot(fig)

# ---------------- CATEGORICAL DISTRIBUTION ---------------- #
st.subheader("📊 Categorical Features Distribution")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    st.write(f"**{col}**")
    fig, ax = plt.subplots()
    sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette="Set2", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------- BOXPLOTS & VIOLIN PLOTS ---------------- #
st.subheader("📦 Boxplots & Violin Plots (Numerical by Category)")
if categorical_cols and numeric_cols:
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            st.write(f"**{num_col} by {cat_col}**")
            fig, axes = plt.subplots(1,2, figsize=(12,4))
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=axes[0], palette="Set3")
            sns.violinplot(x=cat_col, y=num_col, data=df, ax=axes[1], palette="Set2")
            axes[0].tick_params(axis='x', rotation=45)
            axes[1].tick_params(axis='x', rotation=45)
            st.pyplot(fig)

# ---------------- OUTLIER DETECTION ---------------- #
st.subheader("⚠️ Outlier Detection (Numerical Features)")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
    st.write(f"{col} has {len(outliers)} outliers")
    if not outliers.empty:
        st.dataframe(outliers[[col]].head())

# ---------------- CORRELATION HEATMAP ---------------- #
st.subheader("🧩 Correlation Heatmap (Numerical Features)")
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- PAIR PLOTS ---------------- #
st.subheader("🔗 Pair Plot (Numerical Features)")
if len(numeric_cols) >= 2:
    sns.set(style="ticks")
    fig = sns.pairplot(df[numeric_cols])
    st.pyplot(fig)

# ---------------- GROUP COMPARISON ---------------- #
st.subheader("👥 Group Comparison")
if 'DECISION' in df.columns:
    for num_col in numeric_cols:
        st.write(f"**{num_col} by Decision**")
        fig, ax = plt.subplots()
        sns.boxplot(x='DECISION', y=num_col, data=df, palette="Set2", ax=ax)
        st.pyplot(fig)

# ---------------- CUSTOM INSIGHTS ---------------- #
st.subheader("💡 Insights")
if 'CREDIT_SCORE' in df.columns:
    st.write(f"Average Credit Score: {df['CREDIT_SCORE'].mean():.2f}")
    st.write(f"Min Credit Score: {df['CREDIT_SCORE'].min()}")
    st.write(f"Max Credit Score: {df['CREDIT_SCORE'].max()}")

if 'AMT_INCOME_TOTAL' in df.columns:
    st.write(f"Average Income: ₹{df['AMT_INCOME_TOTAL'].mean():,.0f}")
    st.write(f"Income Range: ₹{df['AMT_INCOME_TOTAL'].min():,.0f} - ₹{df['AMT_INCOME_TOTAL'].max():,.0f}")

if 'AGE' in df.columns:
    st.write(f"Average Age: {df['AGE'].mean():.2f}")
    st.write(f"Age Range: {df['AGE'].min()} - {df['AGE'].max()}")
