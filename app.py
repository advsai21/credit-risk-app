import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Credit Risk Analytics Portal", page_icon="💳", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    if "Risk" not in df.columns:
        df["Risk"] = np.where((df["Credit amount"] > 4000) & (df["Duration"] > 24), "bad", "good")
    return df

df = load_data()

st.title("💳 Credit Risk Analytics & Machine Learning Portal")

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Data & Preprocessing", 
    "📊 Exploratory Data Analysis", 
    "🤖 Model Training & Comparison", 
    "🎯 Live Risk Predictor"
])

# TAB 1: PREPROCESSING
with tab1:
    st.header("Dataset Overview & Wrangling")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Features", df.shape[1] - 1)
    c3.metric("Target Variable", "Risk (good / bad)")
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Cleaning Summary")
    st.markdown("""
    * **Missing Imputation:** Missing categorical entries filled with `'Unknown'`.
    * **Encoding:** Categorical variables converted via **One-Hot Encoding**.
    * **Scaling:** Numerical columns (`Age`, `Credit amount`, `Duration`) standard scaled.
    """)

# TAB 2: EDA
with tab2:
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Target Distribution")
        fig_risk = px.pie(df, names="Risk", color="Risk", color_discrete_map={"good": "#2ecc71", "bad": "#e74c3c"}, hole=0.4)
        st.plotly_chart(fig_risk, use_container_width=True)
    with col2:
        st.subheader("Credit Amount vs. Duration")
        fig_scatter = px.scatter(df, x="Duration", y="Credit amount", color="Risk", size="Age", color_discrete_map={"good": "#2ecc71", "bad": "#e74c3c"})
        st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 3: MODEL TRAINING
@st.cache_resource
def train_models(data):
    X = data.drop(columns=["Risk"])
    y = data["Risk"].map({"good": 1, "bad": 0})
    
    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    num_cols = ["Age", "Job", "Credit amount", "Duration"]
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(random_state=42))]).fit(X_train, y_train)
    rf = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=42))]).fit(X_train, y_train)
    
    y_lr = lr.predict(X_test)
    y_rf = rf.predict(X_test)
    
    metrics = pd.DataFrame({
        "Logistic Regression": [accuracy_score(y_test, y_lr), precision_score(y_test, y_lr), recall_score(y_test, y_lr), f1_score(y_test, y_lr)],
        "Random Forest": [accuracy_score(y_test, y_rf), precision_score(y_test, y_rf), recall_score(y_test, y_rf), f1_score(y_test, y_rf)]
    }, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
    
    return lr, rf, metrics, confusion_matrix(y_test, y_lr), confusion_matrix(y_test, y_rf)

lr_model, rf_model, metrics_df, cm_lr, cm_rf = train_models(df)

with tab3:
    st.header("Model Evaluation & Comparison")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("Logistic Regression CM")
        st.plotly_chart(ff.create_annotated_heatmap(cm_lr, x=["Pred Bad", "Pred Good"], y=["Act Bad", "Act Good"], colorscale="Blues"), use_container_width=True)
    with m_col2:
        st.subheader("Random Forest CM")
        st.plotly_chart(ff.create_annotated_heatmap(cm_rf, x=["Pred Bad", "Pred Good"], y=["Act Bad", "Act Good"], colorscale="Greens"), use_container_width=True)

# TAB 4: PREDICTOR
with tab4:
    st.header("Applicant Risk Predictor")
    with st.form("risk_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            age = st.number_input("Age", 18, 80, 30)
            sex = st.selectbox("Sex", df["Sex"].unique())
            housing = st.selectbox("Housing", df["Housing"].unique())
        with p2:
            job = st.number_input("Job Level", 0, 3, 2)
            amount = st.number_input("Credit Amount ($)", 250, 20000, 2500)
            duration = st.number_input("Duration (Months)", 4, 72, 18)
        with p3:
            savings = st.selectbox("Saving Accounts", df["Saving accounts"].dropna().unique())
            checking = st.selectbox("Checking Account", df["Checking account"].dropna().unique())
            purpose = st.selectbox("Purpose", df["Purpose"].unique())
            
        model_type = st.radio("Engine", ["Random Forest", "Logistic Regression"])
        submit = st.form_submit_button("Predict Risk")
        
    if submit:
        sample = pd.DataFrame([{"Age": age, "Sex": sex, "Job": job, "Housing": housing, "Saving accounts": savings, "Checking account": checking, "Credit amount": amount, "Duration": duration, "Purpose": purpose}])
        clf = rf_model if model_type == "Random Forest" else lr_model
        pred = clf.predict(sample)[0]
        prob = clf.predict_proba(sample)[0]
        
        if pred == 1:
            st.success(f"**Low Risk (Good Credit)** — Confidence: {prob[1]*100:.2f}%")
        else:
            st.error(f"**High Risk (Bad Credit)** — Confidence: {prob[0]*100:.2f}%")
