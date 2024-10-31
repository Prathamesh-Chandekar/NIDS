import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Function for detecting anomalies using Isolation Forest
def detect_anomalies(data):
    features = data[['bytes_sent', 'bytes_received', 'packets']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    detector = IsolationForest(contamination=0.1, random_state=42)
    anomalies = detector.fit_predict(scaled_features)
    return anomalies

# Sample data creation function
def create_sample_data():
    data = {
        'bytes_sent': [100, 150, 80, 200, 300, 250, 180, 75, 400, 500],
        'bytes_received': [90, 140, 70, 190, 310, 240, 170, 65, 390, 490],
        'packets': [10, 15, 8, 20, 30, 25, 18, 7, 40, 50]
    }
    return pd.DataFrame(data)

# Streamlit app starts here
st.title("ML Features Analysis")

# Sidebar with options
st.sidebar.header("Options")
page = st.sidebar.selectbox("Choose a Section", ["Current AI/ML Features", "Recommended Enhancements"])

if page == "Current AI/ML Features":
    st.header("Current AI/ML Features in the App")
    
    st.subheader("1. Anomaly Detection")
    st.markdown("""
    - Using **Isolation Forest** algorithm
    - Basic unsupervised learning implementation
    - Limited to detecting statistical outliers
    """)
    
    st.code('''
def detect_anomalies(data):
    features = data[['bytes_sent', 'bytes_received', 'packets']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    detector = IsolationForest(contamination=0.1, random_state=42)
    anomalies = detector.fit_predict(scaled_features)
    return anomalies
    ''', language='python')

    uploaded_file = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"])
    sample_data = create_sample_data()

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())
    else:
        st.write("Sample Data Preview:")
        st.write(sample_data)
        data = sample_data

    if st.button("Run Anomaly Detection"):
        try:
            anomalies = detect_anomalies(data)
            data['Anomaly'] = anomalies
            st.write("Data with Anomalies Flagged:")
            st.write(data)
        except Exception as e:
            st.error(f"Error in detecting anomalies: {e}")
    
    st.download_button(
        label="Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_data.csv",
        mime="text/csv"
    )

    st.subheader("2. Feature Engineering")
    st.markdown("""
    - Basic feature scaling using StandardScaler
    - Simple statistical features only
    - No advanced feature extraction
    """)

    st.subheader("3. Data Preprocessing")
    st.markdown("""
    - Basic data normalization
    - Limited to numerical features
    - No temporal feature extraction
    """)

elif page == "Recommended Enhancements":
    st.header("Recommended AI/ML Enhancements")
    
    st.subheader("1. Advanced Anomaly Detection")
    st.code("""
class NetworkAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.05)
    
    def train(self, X):
        self.isolation_forest.fit(X)
    
    def predict(self, X):
        return self.isolation_forest.predict(X)
    """, language="python")
    
    st.subheader("2. Enhanced Feature Engineering")
    st.markdown("""
    - Implement more complex feature extraction techniques
    - Add domain-specific features for deeper analysis
    """)

    st.subheader("3. Advanced Data Preprocessing")
    st.markdown("""
    - Implement temporal feature extraction for time-series data
    - Explore data augmentation techniques
    """)

st.write("Explore the sections to view current ML features and recommended enhancements.")
