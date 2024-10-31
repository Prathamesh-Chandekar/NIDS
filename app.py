import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json
import base64
from sklearn.metrics import roc_curve, auc

# Modern Streamlit theme settings
st.set_page_config(
    page_title="🛡️ AI-ML Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/network-ids',
        'Report a bug': 'https://github.com/yourusername/network-ids/issues',
        'About': 'A modern AI-ML powered Network Intrusion Detection System'
    }
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        border-color: #FF2B2B;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    div.stActionButton {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "train"

class ModernIDS:
    def __init__(self):
        self.models = {
            '🌳 Random Forest': RandomForestClassifier(n_estimators=100),
            '🧠 Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50)),
            '🚀 XGBoost': XGBClassifier(),
            '🌪️ Gradient Boosting': GradientBoostingClassifier(),
            '🎯 SVM': SVC(probability=True)
        }
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def train(self, X, y, selected_models):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        results = {}
        progress_step = 100 / len(selected_models)
        progress_bar = st.progress(0)
        
        for i, (name, model) in enumerate(
            [(k, v) for k, v in self.models.items() if k in selected_models]
        ):
            with st.spinner(f"Training {name}..."):
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)
                results[name] = {
                    'predictions': y_pred_proba,
                    'model': model
                }
                
                # Calculate feature importance for supported models
                if hasattr(model, 'feature_importances_'):
                    if self.feature_importance is None:
                        self.feature_importance = model.feature_importances_
                    else:
                        self.feature_importance += model.feature_importances_
                
                progress_bar.progress((i + 1) * progress_step)
        
        return X_test, y_test, results
    
    def predict(self, X, selected_models):
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name in selected_models:
            model = self.models[name]
            pred_proba = model.predict_proba(X_scaled)
            predictions[name] = pred_proba
        
        # Ensemble predictions
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        final_pred = np.argmax(ensemble_pred, axis=1)
        confidence = np.max(ensemble_pred, axis=1)
        
        return final_pred, confidence, predictions

def create_sample_data():
    # Generate sample network traffic data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'packet_size': np.random.normal(500, 150, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'port': np.random.randint(1, 65535, n_samples),
        'duration': np.random.exponential(30, n_samples),
        'bytes_transferred': np.random.normal(1000, 300, n_samples),
        'packets_per_second': np.random.normal(50, 15, n_samples),
        'is_encrypted': np.random.choice([0, 1], n_samples),
        'source_entropy': np.random.normal(4, 1, n_samples),
        'malicious': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def render_sidebar():
    st.sidebar.title("🎮 Navigation")
    pages = {
        "🎯 Train Model": "train",
        "🔍 Live Detection": "detect",
        "📊 Analytics": "analytics",
        "ℹ️ About": "about"
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    st.session_state.current_page = pages[selection]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Quick Actions")
    
    if st.sidebar.button("Generate Sample Data"):
        st.session_state.sample_data = create_sample_data()
        st.sidebar.success("Sample data generated! 🎉")

def render_metric_card(title, value, delta=None, suffix=""):
    st.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <h2>{value}{suffix}</h2>
            {f'<p style="color: {"green" if delta >= 0 else "red"}">{"↑" if delta >= 0 else "↓"} {abs(delta)}%</p>' if delta is not None else ''}
        </div>
    """, unsafe_allow_html=True)

def render_train_page():
    st.title("🎯 Train Your Network IDS")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📂 Upload training data (CSV) or use sample data",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif hasattr(st.session_state, 'sample_data'):
            df = st.session_state.sample_data
            st.info("📊 Using generated sample data")
        else:
            st.warning("⚠️ Please upload data or generate sample data")
            return
        
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Feature selection
        target_col = st.selectbox(
            "🎯 Select target column",
            df.columns
        )
        
        feature_cols = st.multiselect(
            "📊 Select features for training",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col]
        )
        
        # Model selection
        st.subheader("🤖 Select Models")
        available_models = [
            '🌳 Random Forest',
            '🧠 Neural Network',
            '🚀 XGBoost',
            '🌪️ Gradient Boosting',
            '🎯 SVM'
        ]
        selected_models = st.multiselect(
            "Choose models for ensemble",
            available_models,
            default=['🌳 Random Forest', '🧠 Neural Network']
        )
    
    with col2:
        st.subheader("📊 Data Statistics")
        st.write(df[feature_cols].describe())
    
    if st.button("🚀 Train Models", use_container_width=True):
        X = df[feature_cols]
        y = df[target_col]
        
        st.session_state.model = ModernIDS()
        with st.spinner("🔧 Training models..."):
            X_test, y_test, results = st.session_state.model.train(
                X, y, selected_models
            )
            st.session_state.trained = True
            st.session_state.feature_cols = feature_cols
        
        st.success("✨ Models trained successfully!")
        
        # Show performance metrics
        st.subheader("📈 Model Performance")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "📊 Performance Metrics",
            "🎯 Feature Importance",
            "📈 ROC Curves"
        ])
        
        with tab1:
            display_performance_metrics(results, X_test, y_test)
        
        with tab2:
            if st.session_state.model.feature_importance is not None:
                display_feature_importance(
                    feature_cols,
                    st.session_state.model.feature_importance
                )
        
        with tab3:
            display_roc_curves(results, X_test, y_test)

def display_performance_metrics(results, X_test, y_test):
    metrics_df = pd.DataFrame()
    
    for name, result in results.items():
        y_pred = np.argmax(result['predictions'], axis=1)
        accuracy = np.mean(y_pred == y_test)
        metrics_df.loc[name, 'Accuracy'] = accuracy * 100
    
    fig = px.bar(
        metrics_df,
        title="Model Accuracy Comparison",
        labels={'value': 'Accuracy (%)', 'index': 'Model'},
        color_discrete_sequence=['#FF4B4B']
    )
    st.plotly_chart(fig, use_container_width=True)

def display_feature_importance(feature_cols, importance):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance",
        color_discrete_sequence=['#FF4B4B']
    )
    st.plotly_chart(fig, use_container_width=True)

def display_roc_curves(results, X_test, y_test):
    fig = go.Figure()
    
    for name, result in results.items():
        y_pred_proba = result['predictions'][:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{name} (AUC = {auc_score:.3f})"
            )
        )
    
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(dash='dash'),
            name='Random'
        )
    )
    
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_detect_page():
    st.title("🔍 Live Network Traffic Detection")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train the model first!")
        return
    
    uploaded_file = st.file_uploader(
        "📂 Upload network traffic data for analysis",
        type=['csv']
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if st.button("🔍 Analyze Traffic", use_container_width=True):
            predictions, confidence, model_predictions = st.session_state.model.predict(
                df[st.session_state.feature_cols],
                st.session_state.model.models.keys()
            )
            
            # Display results in modern cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                render_metric_card(
                    "Total Traffic Analyzed",
                    len(predictions),
                    suffix=" packets"
                )
            
            with col2:
                render_metric_card(
                    "Suspicious Traffic",
                    f"{(predictions == 1).mean():.1%}"
                )
            
            with col3:
                render_metric_card(
                    "Average Confidence",
                    f"{confidence.mean():.1%}"
                )
            
            # Create visualization tabs
            tab1, tab2 = st.tabs([
                "📊 Detection Results",
                "📈 Confidence Analysis"
            ])
            
            with tab1:
                display_detection_results(predictions, confidence, df)
            
            with tab2:
                display_confidence_analysis(predictions, confidence)

def display_detection_results(predictions, confidence, df):
    results_df = pd.DataFrame({
        'Status': np.where(predictions == 1, '⚠️ Suspicious', '✅ Normal'),
        'Confidence': confidence,
        'Timestamp': pd.date_range(
            start=datetime.now(),
            periods=len(predictions),
            freq='S'
        )
    })
    
    # Plot traffic status over time
    fig = px.line(
        results_df,
        x='Timestamp',
        y='Confidence',
        color='Status',
        title="Network Traffic Analysis Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed results
    st.dataframe(
        results_df.style.background_gradient(
            subset=['Confidence'],
            cmap='RdYlGn_r'
        ),
        use_container_width=True
    )

def display_confidence_analysis(predictions, confidence):
    st.subheader("📈 Confidence Analysis")
    
    # Analyze confidence distribution
    confidence_df = pd.DataFrame({
        'Confidence': confidence,
        'Status': np.where(predictions == 1, 'Suspicious', 'Normal')
    })
    
    # Plot confidence distribution for normal and suspicious traffic
    fig = px.histogram(
        confidence_df,
        x='Confidence',
        color='Status',
        barmode='overlay',
        title="Confidence Distribution by Traffic Status",
        color_discrete_sequence=['#FF4B4B', '#00CC96']
    )
    fig.update_layout(xaxis_title="Confidence", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    # Display statistics
    st.write("### Confidence Summary")
    st.write(confidence_df.groupby('Status')['Confidence'].describe())
