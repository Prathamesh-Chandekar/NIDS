import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

class StreamlitIDS:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(50, 25))
        self.scaler = StandardScaler()
        
    def train(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train models with progress bar
        progress_bar = st.progress(0)
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        progress_bar.progress(50)
        
        # Train Neural Network
        self.nn_model.fit(X_train, y_train)
        progress_bar.progress(100)
        
        # Get predictions
        rf_pred = self.rf_model.predict_proba(X_test)
        nn_pred = self.nn_model.predict_proba(X_test)
        ensemble_pred = 0.6 * rf_pred + 0.4 * nn_pred
        y_pred = np.argmax(ensemble_pred, axis=1)
        
        return X_test, y_test, y_pred
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict_proba(X_scaled)
        nn_pred = self.nn_model.predict_proba(X_scaled)
        
        # Ensemble predictions
        ensemble_pred = 0.6 * rf_pred + 0.4 * nn_pred
        predictions = np.argmax(ensemble_pred, axis=1)
        confidence = np.max(ensemble_pred, axis=1)
        
        return predictions, confidence

def main():
    st.title("🛡️ Network Intrusion Detection System")
    st.write("Upload network traffic data to detect potential security threats")
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload training data (CSV)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file)
        
        # Display data overview
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Sample of uploaded data:")
            st.dataframe(df.head())
        
        with col2:
            st.write("Data Statistics:")
            st.dataframe(df.describe())
        
        # Feature selection
        st.subheader("Feature Selection")
        target_col = st.selectbox(
            "Select target column (labels)",
            df.columns
        )
        
        feature_cols = st.multiselect(
            "Select features for training",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col]
        )
        
        # Train model button
        if st.button("Train Model"):
            try:
                X = df[feature_cols]
                y = df[target_col]
                
                # Initialize and train model
                st.session_state.model = StreamlitIDS()
                
                with st.spinner("Training model..."):
                    X_test, y_test, y_pred = st.session_state.model.train(X, y)
                    st.session_state.trained = True
                
                # Display results
                st.success("Model trained successfully!")
                
                # Show model performance
                st.subheader("Model Performance")
                
                # Confusion matrix
                confusion_matrix = pd.crosstab(
                    y_test, y_pred,
                    rownames=['Actual'],
                    colnames=['Predicted']
                )
                
                fig = px.imshow(
                    confusion_matrix,
                    labels=dict(x="Predicted", y="Actual"),
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    # Live prediction section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Prediction")
    
    prediction_file = st.sidebar.file_uploader(
        "Upload data for prediction",
        type=['csv']
    )
    
    if prediction_file is not None and st.session_state.trained:
        pred_df = pd.read_csv(prediction_file)
        
        if st.sidebar.button("Run Prediction"):
            with st.spinner("Analyzing network traffic..."):
                # Make predictions
                predictions, confidence = st.session_state.model.predict(
                    pred_df[feature_cols]
                )
                
                # Display results
                st.subheader("Prediction Results")
                
                results_df = pd.DataFrame({
                    'Prediction': predictions,
                    'Confidence': confidence
                })
                
                # Add traffic status
                results_df['Status'] = np.where(
                    predictions == 1,
                    '⚠️ Suspicious',
                    '✅ Normal'
                )
                
                # Display results with color coding
                st.dataframe(
                    results_df.style.background_gradient(
                        subset=['Confidence'],
                        cmap='RdYlGn_r'
                    )
                )
                
                # Plot confidence distribution
                fig = px.histogram(
                    results_df,
                    x='Confidence',
                    color='Status',
                    title='Prediction Confidence Distribution'
                )
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
