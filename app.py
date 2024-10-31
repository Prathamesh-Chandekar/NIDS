import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import altair as alt

# Title and Description
st.title("CyberGuard - Network Security ML Dashboard")
st.markdown("""
CyberGuard is a network security ML dashboard designed to help identify anomalies and security alerts in network data. 
Upload your network data to detect unusual patterns and stay on top of potential security threats.
""")

# Sidebar Options
st.sidebar.header("Dashboard Options")
page = st.sidebar.radio("Select a Section", ["Data Overview", "Anomaly Detection", "Security Alerts", "Visualizations"])

# Sample Data Function
def create_sample_data():
    data = {
        'bytes_sent': np.random.randint(50, 1000, size=100),
        'bytes_received': np.random.randint(50, 1000, size=100),
        'packets': np.random.randint(1, 100, size=100)
    }
    return pd.DataFrame(data)

# Anomaly Detection Function
def detect_anomalies(data):
    features = data[['bytes_sent', 'bytes_received', 'packets']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    detector = IsolationForest(contamination=0.1, random_state=42)
    data['Anomaly'] = detector.fit_predict(scaled_features)
    data['Alert'] = data['Anomaly'].apply(lambda x: 'Yes' if x == -1 else 'No')
    return data

# Load Sample Data
sample_data = create_sample_data()

# Data Upload Section
if page == "Data Overview":
    st.subheader("1. Data Upload & Overview")
    uploaded_file = st.file_uploader("Upload CSV for Analysis", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", data.head())
    else:
        st.write("Sample Data Preview:")
        st.write(sample_data)
        data = sample_data

# Anomaly Detection Section
elif page == "Anomaly Detection":
    st.subheader("2. Run Anomaly Detection")
    uploaded_file = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = sample_data
    
    if st.button("Run Anomaly Detection"):
        data_with_anomalies = detect_anomalies(data)
        st.write("Data with Anomalies Flagged:")
        st.write(data_with_anomalies)
        
        # Option to download the flagged data
        st.download_button(
            label="Download Flagged Data",
            data=data_with_anomalies.to_csv(index=False),
            file_name="flagged_data.csv",
            mime="text/csv"
        )

# Security Alerts Section
elif page == "Security Alerts":
    st.subheader("3. Security Alerts")
    
    data_with_anomalies = detect_anomalies(sample_data)
    alerts = data_with_anomalies[data_with_anomalies['Alert'] == 'Yes']
    
    if not alerts.empty:
        st.write("Anomalies Detected - Potential Security Alerts:")
        st.write(alerts)
    else:
        st.success("No anomalies detected in the sample data.")

# Visualization Section
elif page == "Visualizations":
    st.subheader("4. Network Traffic Visualizations")
    st.markdown("Visualize data trends to gain insights into network traffic patterns.")
    
    data_with_anomalies = detect_anomalies(sample_data)

    # Bytes Sent vs Bytes Received
    chart = alt.Chart(data_with_anomalies).mark_circle(size=60).encode(
        x='bytes_sent',
        y='bytes_received',
        color=alt.condition(
            alt.datum.Alert == 'Yes',
            alt.value('red'),  # Anomalies
            alt.value('blue')  # Normal
        ),
        tooltip=['bytes_sent', 'bytes_received', 'packets', 'Alert']
    ).properties(
        title='Network Traffic: Bytes Sent vs. Bytes Received'
    )
    
    st.altair_chart(chart, use_container_width=True)

    # Packets Distribution
    st.write("Packets Distribution:")
    packets_chart = alt.Chart(data_with_anomalies).mark_bar().encode(
        x=alt.X('packets', bin=alt.Bin(maxbins=30)),
        y='count()',
        color=alt.condition(
            alt.datum.Alert == 'Yes',
            alt.value('red'),
            alt.value('blue')
        )
    )
    
    st.altair_chart(packets_chart, use_container_width=True)
