import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# PAGE CONFIG & CUSTOM CSS (PREMIUM THEMING)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk AI Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Custom Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Input Controls */
    .stSelectbox, .stNumberInput {
        border-radius: 8px;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #3d82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_dict_only=True)

# ---------------------------------------------------------
# DATA PIPELINE
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    if "Risk" not in df.columns:
        df["Risk"] = np.where((df["Credit amount"] > 4000) & (df["Duration"] > 24), "bad", "good")
    return df

df = load_data()

# PLOTLY GLASS THEME HELPER
PLOTLY_THEME = "plotly_dark"
COLOR_GOOD = "#10b981"
COLOR_BAD = "#ef4444"

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.markdown('<p class="main-title">💳 Credit Risk AI Intelligence Hub</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Data Analytics, Interactive Visualizations & Predictive ML Workbench</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & FILTERS
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric-reflection/100/bank-cards.png", width=70)
st.sidebar.title("App Navigation")
selected_tab = st.sidebar.radio("Go to:", [
    "📊 Executive Summary", 
    "📈 Exploratory Analytics", 
    "🤖 ML Models & Evaluation", 
    "⚡ Live Applicant Risk Scoring"
])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Global EDA Filters")
filter_sex = st.sidebar.multiselect("Gender", options=df["Sex"].unique(), default=df["Sex"].unique())
filter_housing = st.sidebar.multiselect("Housing", options=df["Housing"].unique(), default=df["Housing"].unique())

filtered_df = df[(df["Sex"].isin(filter_sex)) & (df["Housing"].isin(filter_housing))]

# ---------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------
if selected_tab == "📊 Executive Summary":
    st.markdown('### 📌 Dataset & Portfolio KPI Metrics')
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Loan Applicants", f"{df.shape[0]:,}")
    kpi2.metric("Low Risk Ratio", f"{(df['Risk'] == 'good').mean()*100:.1f}%")
    kpi3.metric("Avg Credit Amount", f"${df['Credit amount'].mean():,.0f}")
    kpi4.metric("Avg Duration", f"{df['Duration'].mean():.0f} Months")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1.2, 1])
    
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Data Overview & Inspection")
        st.dataframe(df.head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Preprocessing Pipeline Summary")
        st.markdown("""
        * **Categorical Imputation:** Standardized missing values in `Saving accounts` & `Checking account` with `'Unknown'`.
        * **Categorical Encoding:** Leveraged `OneHotEncoder` with `handle_unknown='ignore'`.
        * **Feature Scaling:** Applied `StandardScaler` to continuous fields (`Age`, `Credit amount`, `Duration`).
        * **Target Framing:** Standardized risk classification (`good`: 1, `bad`: 0).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: EXPLORATORY ANALYTICS
# ---------------------------------------------------------
elif selected_tab == "📈 Exploratory Analytics":
    st.markdown('### 🔍 Exploratory Data Analysis Dashboard')
    
    row1_1, row1_2 = st.columns(2)
    
    with row1_1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Risk Share Breakdown")
        fig_pie = px.pie(
            filtered_df, names="Risk", color="Risk",
            color_discrete_map={"good": COLOR_GOOD, "bad": COLOR_BAD},
            hole=0.55, template=PLOTLY_THEME
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with row1_2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Credit Amount vs Duration")
        fig_scatter = px.scatter(
            filtered_df, x="Duration", y="Credit amount", color="Risk", size="Age",
            color_discrete_map={"good": COLOR_GOOD, "bad": COLOR_BAD},
            template=PLOTLY_THEME, opacity=0.8
        )
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row2_1, row2_2 = st.columns(2)
    
    with row2_1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Risk Distribution by Purpose")
        fig_bar = px.histogram(
            filtered_df, x="Purpose", color="Risk", barmode="group",
            color_discrete_map={"good": COLOR_GOOD, "bad": COLOR_BAD},
            template=PLOTLY_THEME
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with row2_2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Numeric Features Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()
        fig_heatmap = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", template=PLOTLY_THEME)
        fig_heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: MODEL EVALUATION
# ---------------------------------------------------------
elif selected_tab == "🤖 ML Models & Evaluation":
    st.markdown('### 🤖 Model Training & Benchmarking Engine')
    
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
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Performance Comparison")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#10b981"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Logistic Regression CM")
        fig_cm_lr = px.imshow(cm_lr, x=["Bad Risk", "Good Risk"], y=["Bad Risk", "Good Risk"], text_auto=True, color_continuous_scale="Blues", template=PLOTLY_THEME)
        fig_cm_lr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm_lr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with m_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Random Forest CM")
        fig_cm_rf = px.imshow(cm_rf, x=["Bad Risk", "Good Risk"], y=["Bad Risk", "Good Risk"], text_auto=True, color_continuous_scale="Greens", template=PLOTLY_THEME)
        fig_cm_rf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: LIVE PREDICTOR
# ---------------------------------------------------------
elif selected_tab == "⚡ Live Applicant Risk Scoring":
    st.markdown('### ⚡ Interactive Risk Scoring Form')
    
    # Train instances
    lr_model, rf_model, _, _, _ = train_models(df)
    
    with st.form("risk_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("##### 👤 Personal Profile")
            age = st.number_input("Age", 18, 80, 30)
            sex = st.selectbox("Sex", df["Sex"].unique())
            housing = st.selectbox("Housing", df["Housing"].unique())
            
        with p2:
            st.markdown("##### 💰 Financial Terms")
            job = st.number_input("Job Level (0 - 3)", 0, 3, 2)
            amount = st.number_input("Credit Amount ($)", 250, 20000, 2500)
            duration = st.number_input("Loan Duration (Months)", 4, 72, 18)
            
        with p3:
            st.markdown("##### 🏦 Account & Purpose")
            savings = st.selectbox("Saving Accounts", df["Saving accounts"].dropna().unique())
            checking = st.selectbox("Checking Account", df["Checking account"].dropna().unique())
            purpose = st.selectbox("Loan Purpose", df["Purpose"].unique())
            
        st.markdown("---")
        model_type = st.radio("Select Prediction Engine:", ["Random Forest Classifier", "Logistic Regression"], horizontal=True)
        
        submit = st.form_submit_button("⚡ Predict Credit Risk")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if submit:
        sample = pd.DataFrame([{"Age": age, "Sex": sex, "Job": job, "Housing": housing, "Saving accounts": savings, "Checking account": checking, "Credit amount": amount, "Duration": duration, "Purpose": purpose}])
        clf = rf_model if model_type == "Random Forest Classifier" else lr_model
        pred = clf.predict(sample)[0]
        prob = clf.predict_proba(sample)[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if pred == 1:
            st.balloons()
            st.success(f"### 🎉 Result: LOW RISK (Good Credit Approval)\n**Model Confidence:** {prob[1]*100:.2f}%")
        else:
            st.error(f"### ⚠️ Result: HIGH RISK (Bad Credit Alert)\n**Model Confidence:** {prob[0]*100:.2f}%")
