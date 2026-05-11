import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1f36; }
    [data-testid="stSidebar"] * { color: #e0e6ff !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #16213e);
        border: 1px solid #2d5a8e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #90caf9; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #1565c0, #0d47a1);
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 16px 0 10px 0;
    }
    .predict-yes {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border: 2px solid #66bb6a;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }
    .predict-no {
        background: linear-gradient(135deg, #7f0000, #b71c1c);
        border: 2px solid #ef5350;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load & cache data
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('bank.csv')
    return df

@st.cache_resource
def train_models(df):
    data = df.copy()
    data['deposit'] = data['deposit'].map({'yes': 1, 'no': 0})
    data = pd.get_dummies(data, drop_first=True)
    data.fillna(data.median(numeric_only=True), inplace=True)

    X = data.drop('deposit', axis=1)
    y = data['deposit']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_r, y_train_r = smote.fit_resample(X_train_s, y_train)

    # 1. Logistic Regression
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_r, y_train_r)
    log_pred = log_model.predict(X_test_s)

    # 2. Polynomial Logistic Regression
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_r)
    X_test_poly  = poly.transform(X_test_s)
    poly_log_model = LogisticRegression(max_iter=1000, C=0.01, solver='saga', random_state=42)
    poly_log_model.fit(X_train_poly, y_train_r)
    poly_log_pred = poly_log_model.predict(X_test_poly)

    # 3. Linear Regression (single variable)
    X_train_single = X_train_r[:, 0].reshape(-1, 1)
    X_test_single  = X_test_s[:, 0].reshape(-1, 1)
    lin_single = LinearRegression()
    lin_single.fit(X_train_single, y_train_r)
    pred_single = (lin_single.predict(X_test_single) >= 0.5).astype(int)

    # 4. Linear Regression (multi variable)
    lin_multi = LinearRegression()
    lin_multi.fit(X_train_r, y_train_r)
    pred_multi = (lin_multi.predict(X_test_s) >= 0.5).astype(int)

    # 5. Polynomial Linear Regression
    poly_lin = PolynomialFeatures(degree=2)
    X_train_poly_lin = poly_lin.fit_transform(X_train_r)
    X_test_poly_lin  = poly_lin.transform(X_test_s)
    lin_poly = LinearRegression()
    lin_poly.fit(X_train_poly_lin, y_train_r)
    pred_poly_lin = (lin_poly.predict(X_test_poly_lin) >= 0.5).astype(int)

    results = {
        "Logistic Regression": {
            "model": log_model, "pred": log_pred,
            "acc": accuracy_score(y_test, log_pred),
            "report": classification_report(y_test, log_pred, output_dict=True),
            "cm": confusion_matrix(y_test, log_pred),
            "train_acc": accuracy_score(y_train_r, log_model.predict(X_train_r)),
        },
        "Polynomial Logistic": {
            "model": poly_log_model, "pred": poly_log_pred,
            "acc": accuracy_score(y_test, poly_log_pred),
            "report": classification_report(y_test, poly_log_pred, output_dict=True),
            "cm": confusion_matrix(y_test, poly_log_pred),
            "train_acc": accuracy_score(y_train_r, poly_log_model.predict(X_train_poly)),
        },
        "Linear Regression (Single)": {
            "pred": pred_single,
            "acc": accuracy_score(y_test, pred_single),
            "report": classification_report(y_test, pred_single, output_dict=True),
            "cm": confusion_matrix(y_test, pred_single),
        },
        "Linear Regression (Multi)": {
            "pred": pred_multi,
            "acc": accuracy_score(y_test, pred_multi),
            "report": classification_report(y_test, pred_multi, output_dict=True),
            "cm": confusion_matrix(y_test, pred_multi),
        },
        "Polynomial Linear": {
            "pred": pred_poly_lin,
            "acc": accuracy_score(y_test, pred_poly_lin),
            "report": classification_report(y_test, pred_poly_lin, output_dict=True),
            "cm": confusion_matrix(y_test, pred_poly_lin),
        },
    }

    artifacts = {
        "scaler": scaler,
        "poly_features": poly,
        "feature_names": list(X.columns),
        "X_test": X_test_s,
        "y_test": y_test,
    }
    return results, artifacts, data

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Bank Marketing ML")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dataset Overview", "🔍 EDA", "🤖 Model Results", "🔮 Predict"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** Bank Marketing")
    st.markdown("**Target:** Term Deposit (yes/no)")
    st.markdown("**Models:** 5 regression variants")

# ─────────────────────────────────────────
# Load
# ─────────────────────────────────────────
df = load_data()

with st.spinner("🔄 Training models..."):
    results, artifacts, df_encoded = train_models(df)

# ═══════════════════════════════════════════════════════
# PAGE 1 – Dataset Overview
# ═══════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("**Bank Marketing Dataset** — predicting whether a client will subscribe to a term deposit.")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [len(df), df.shape[1], df['deposit'].value_counts()['yes'], df['deposit'].value_counts()['no']],
        ["Total Records", "Features", "Subscribed (Yes)", "Not Subscribed (No)"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val:,}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown('<div class="section-header">📋 Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    with c_right:
        st.markdown('<div class="section-header">📐 Feature Info</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Nulls": [df[c].isnull().sum() for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, height=350)

    st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["🎯 Target Distribution", "📊 Feature Distributions", "🔗 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            counts = df['deposit'].value_counts()
            colors = ['#4fc3f7', '#ef5350']
            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{bar.get_height():,}', ha='center', color='white', fontsize=11)
            ax.set_title('Deposit Distribution', color='white', fontsize=13)
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_edgecolor('#333')
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.pie(counts.values, labels=counts.index, colors=colors,
                   autopct='%1.1f%%', textprops={'color': 'white'}, startangle=90,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 1.2})
            ax.set_title('Deposit %', color='white', fontsize=13)
            st.pyplot(fig)

        st.markdown("---")
        cat_col = st.selectbox("Choose a categorical feature vs. Deposit:",
                               ['job', 'marital', 'education', 'contact', 'poutcome'])
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        cross = pd.crosstab(df[cat_col], df['deposit'], normalize='index') * 100
        cross.plot(kind='bar', ax=ax, color=['#ef5350', '#4fc3f7'], edgecolor='white', linewidth=0.5)
        ax.set_title(f'{cat_col.title()} vs Deposit (%)', color='white', fontsize=13)
        ax.tick_params(colors='white', axis='both')
        ax.set_xlabel('', color='white')
        ax.legend(['No', 'Yes'], facecolor='#1a1f36', labelcolor='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

    with tab2:
        num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        selected = st.selectbox("Select numeric feature:", num_cols)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            for dep, color in [('yes', '#4fc3f7'), ('no', '#ef5350')]:
                df[df['deposit'] == dep][selected].hist(
                    ax=ax, bins=30, alpha=0.6, color=color, label=dep, edgecolor='none')
            ax.set_title(f'{selected} Distribution by Deposit', color='white', fontsize=12)
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1a1f36', labelcolor='white')
            for spine in ax.spines.values(): spine.set_edgecolor('#333')
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            data_yes = df[df['deposit'] == 'yes'][selected]
            data_no  = df[df['deposit'] == 'no'][selected]
            bp = ax.boxplot([data_no, data_yes], labels=['No', 'Yes'],
                            patch_artist=True,
                            boxprops=dict(facecolor='#1e3a5f', color='white'),
                            medianprops=dict(color='#4fc3f7', linewidth=2),
                            whiskerprops=dict(color='white'),
                            capprops=dict(color='white'),
                            flierprops=dict(color='#ef5350', markeredgecolor='#ef5350'))
            bp['boxes'][0].set_facecolor('#ef5350')
            bp['boxes'][1].set_facecolor('#4fc3f7')
            ax.set_title(f'{selected} Boxplot', color='white', fontsize=12)
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_edgecolor('#333')
            st.pyplot(fig)

    with tab3:
        num_df = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']].copy()
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, linewidths=0.5, linecolor='#1a1f36',
                    annot_kws={'color': 'white', 'size': 9})
        ax.tick_params(colors='white', labelsize=9)
        ax.set_title('Feature Correlation Heatmap', color='white', fontsize=13, pad=12)
        plt.tight_layout()
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════
# PAGE 3 – Model Results
# ═══════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.title("🤖 Model Evaluation Results")

    # Summary table
    st.markdown('<div class="section-header">📊 Accuracy Comparison</div>', unsafe_allow_html=True)
    summary = pd.DataFrame([
        {"Model": name, "Test Accuracy": f"{v['acc']*100:.2f}%",
         "Precision (avg)": f"{v['report']['weighted avg']['precision']*100:.2f}%",
         "Recall (avg)": f"{v['report']['weighted avg']['recall']*100:.2f}%",
         "F1 (avg)": f"{v['report']['weighted avg']['f1-score']*100:.2f}%"}
        for name, v in results.items()
    ])
    st.dataframe(summary, use_container_width=True)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    names = list(results.keys())
    accs  = [results[n]['acc'] * 100 for n in names]
    colors = ['#4fc3f7' if a == max(accs) else '#2d5a8e' for a in accs]
    bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color='white', fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Accuracy %', color='white')
    ax.set_title('Model Accuracy Comparison', color='white', fontsize=13)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')
    plt.tight_layout()
    st.pyplot(fig)

    # Per-model details
    st.markdown('<div class="section-header">🔬 Detailed Results per Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Choose a model:", list(results.keys()))
    r = results[selected_model]

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-value">{r['acc']*100:.2f}%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-value">{r['report']['weighted avg']['precision']*100:.2f}%</div>
        <div class="metric-label">Precision (weighted)</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-value">{r['report']['weighted avg']['f1-score']*100:.2f}%</div>
        <div class="metric-label">F1 Score (weighted)</div>
    </div>""", unsafe_allow_html=True)

    if 'train_acc' in r:
        diff = r['train_acc'] - r['acc']
        status = "⚠️ Overfitting" if diff > 0.05 else "✅ Model is Good"
        st.info(f"Train Accuracy: {r['train_acc']*100:.2f}%  |  {status}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                linewidths=0.5, linecolor='#1a1f36',
                annot_kws={'color': 'white', 'size': 12})
    ax.set_title('Confusion Matrix', color='white', fontsize=12)
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('Actual', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════
# PAGE 4 – Predict
# ═══════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Predict Term Deposit")
    st.markdown("Enter client details below and the model will predict if they will subscribe.")

    model_choice = st.selectbox(
        "Choose Prediction Model:",
        ["Logistic Regression", "Polynomial Logistic"]
    )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 18, 95, 35)
        job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management',
                                    'retired', 'blue-collar', 'unemployed', 'entrepreneur',
                                    'housemaid', 'unknown', 'self-employed', 'student'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        education = st.selectbox("Education", ['secondary', 'tertiary', 'primary', 'unknown'])
        default = st.selectbox("Has Credit Default?", ['no', 'yes'])

    with c2:
        balance = st.number_input("Account Balance (€)", -7000, 82000, 1000, step=100)
        housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
        contact = st.selectbox("Contact Type", ['unknown', 'cellular', 'telephone'])
        month = st.selectbox("Last Contact Month", ['jan','feb','mar','apr','may','jun',
                                                     'jul','aug','sep','oct','nov','dec'])

    with c3:
        day = st.slider("Day of Month", 1, 31, 15)
        duration = st.slider("Call Duration (seconds)", 0, 3900, 200)
        campaign = st.slider("Contacts in Campaign", 1, 50, 3)
        pdays = st.number_input("Days Since Last Contact (-1 = never)", -1, 900, -1)
        previous = st.slider("Previous Contacts", 0, 60, 0)
        poutcome = st.selectbox("Previous Outcome", ['unknown', 'other', 'failure', 'success'])

    if st.button("🚀 Predict", use_container_width=True):
        input_dict = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
            'contact': contact, 'day': day, 'month': month, 'duration': duration,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'deposit': 'no'
        }
        input_df = pd.DataFrame([input_dict])
        input_df['deposit'] = input_df['deposit'].map({'yes': 1, 'no': 0})
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        expected_cols = artifacts['feature_names']
        for col in expected_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[expected_cols]

        scaler = artifacts['scaler']
        input_scaled = scaler.transform(input_encoded)

        model = results[model_choice]['model']
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="predict-yes">
                <h2 style="color:#a5d6a7; margin:0">✅ Will Subscribe</h2>
                <p style="color:#e8f5e9; font-size:1.1rem; margin-top:8px">
                    This client is likely to subscribe to a term deposit.<br>
                    <b>Confidence: {proba[1]*100:.1f}%</b>
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-no">
                <h2 style="color:#ffcdd2; margin:0">❌ Will NOT Subscribe</h2>
                <p style="color:#ffebee; font-size:1.1rem; margin-top:8px">
                    This client is unlikely to subscribe to a term deposit.<br>
                    <b>Confidence: {proba[0]*100:.1f}%</b>
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        bars = ax.barh(['No', 'Yes'], [proba[0]*100, proba[1]*100],
                       color=['#ef5350', '#4fc3f7'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, [proba[0]*100, proba[1]*100]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='white', fontsize=11)
        ax.set_xlim(0, 110)
        ax.set_xlabel('Probability %', color='white')
        ax.set_title('Prediction Probabilities', color='white', fontsize=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        plt.tight_layout()
        st.pyplot(fig)
