import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import altair as alt
from datetime import datetime, timedelta

# Title and Description with Emojis
st.title("🛡️ CyberGuard - Enhanced Network Security ML Dashboard")
st.markdown("""
CyberGuard is a network security ML dashboard that identifies anomalies, security alerts, and potential threats in network data. This updated version includes threat types, timestamps, and risk levels for each alert.
""")

# Sidebar Options
st.sidebar.header("⚙️ Dashboard Options")
page = st.sidebar.radio("Select a Section", ["📊 Data Overview", "🧪 Anomaly Detection", "🚨 Security Alerts", "🔍 Malware Type Dashboard", "📈 Visualizations"])

# Sample Data Creation with Enhanced Columns
def create_sample_data():
    malware_types = [None, "Trojan", "Spyware", "Ransomware", "Phishing"]
    risk_levels = ["Low", "Medium", "High"]
    timestamps = [datetime.now() - timedelta(minutes=i*10) for i in range(100)]
    
    data = {
        'timestamp': np.random.choice(timestamps, size=100),
        'bytes_sent': np.random.randint(50, 1000, size=100),
        'bytes_received': np.random.randint(50, 1000, size=100),
        'packets': np.random.randint(1, 100, size=100),
        'source_ip': np.random.choice(["192.168.0.1", "192.168.0.2", "10.0.0.1"], size=100),
        'destination_ip': np.random.choice(["192.168.0.10", "10.0.0.5", "172.16.0.1"], size=100),
        'malware_type': np.random.choice(malware_types, size=100, p=[0.85, 0.05, 0.05, 0.03, 0.02]),
        'risk_level': np.random.choice(risk_levels, size=100, p=[0.6, 0.3, 0.1])
    }
    return pd.DataFrame(data)

# Anomaly Detection with Enhanced Data
def detect_anomalies(data):
    features = data[['bytes_sent', 'bytes_received', 'packets']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    detector = IsolationForest(contamination=0.1, random_state=42)
    data['Anomaly'] = detector.fit_predict(scaled_features)
    data['Alert'] = data['Anomaly'].apply(lambda x: '🚨 Yes' if x == -1 else '✅ No')
    
    # Confidence score
    data['Confidence_Score'] = np.random.uniform(70, 99, data.shape[0])  # Simulating confidence scores between 70-99%
    
    return data

# Load Sample Data with Enhanced Details
sample_data = create_sample_data()

# Center-align tables using CSS
st.markdown(
    """
    <style>
    .dataframe {
        margin-left: auto;
        margin-right: auto;
    }
    .card {
        background-color: #f9f9f9;
        padding: 10px;
        margin: 5px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data Upload Section
if page == "📊 Data Overview":
    st.subheader("1. Data Upload & Overview")
    uploaded_file = st.file_uploader("📁 Upload CSV for Analysis", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### 🔍 Uploaded Data:", data.head().style.set_properties(**{'text-align': 'center'}))
    else:
        st.write("### 🔍 Sample Data Preview:")
        st.write(sample_data.style.set_properties(**{'text-align': 'center'}))
        data = sample_data

# Anomaly Detection Section
elif page == "🧪 Anomaly Detection":
    st.subheader("2. Run Anomaly Detection")
    uploaded_file = st.file_uploader("📁 Upload CSV for Anomaly Detection", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = sample_data
    
    if st.button("🛠️ Run Anomaly Detection"):
        data_with_anomalies = detect_anomalies(data)
        st.write("### 🕵️ Data with Anomalies Flagged:")
        st.write(data_with_anomalies.style.set_properties(**{'text-align': 'center'}))
        
        st.download_button(
            label="💾 Download Flagged Data",
            data=data_with_anomalies.to_csv(index=False),
            file_name="flagged_data.csv",
            mime="text/csv"
        )

# Security Alerts Section
elif page == "🚨 Security Alerts":
    st.subheader("3. Security Alerts")
    
    data_with_anomalies = detect_anomalies(sample_data)
    alerts = data_with_anomalies[data_with_anomalies['Alert'] == '🚨 Yes']
    
    if not alerts.empty:
        st.write("### ⚠️ Anomalies Detected - Potential Security Alerts:")
        st.write(alerts[['timestamp', 'source_ip', 'destination_ip', 'bytes_sent', 'bytes_received', 'packets', 'malware_type', 'risk_level', 'Alert']].style.set_properties(**{'text-align': 'center'}))
    else:
        st.success("✅ No anomalies detected in the sample data.")

# Malware Type Dashboard
elif page == "🔍 Malware Type Dashboard":
    st.subheader("4. Malware Type Dashboard")
    data_with_anomalies = detect_anomalies(sample_data)
    
    malware_summary = data_with_anomalies[data_with_anomalies['malware_type'].notna()]

    # Display Cards for details
    for _, row in malware_summary.iterrows():
        st.markdown(
            f"""
            <div class="card">
                <h3>🚨 {row['malware_type']} Detected</h3>
                <p>Source IP: {row['source_ip']}</p>
                <p>Destination IP: {row['destination_ip']}</p>
                <p>Risk Level: {row['risk_level']}</p>
                <p>Confidence Score: {row['Confidence_Score']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True
        )
    
# Visualization Section
elif page == "📈 Visualizations":
    st.subheader("5. Network Traffic Visualizations")
    st.markdown("📊 Visualize data trends to gain insights into network traffic patterns.")
    
    data_with_anomalies = detect_anomalies(sample_data)

    # Bytes Sent vs Bytes Received
    chart = alt.Chart(data_with_anomalies).mark_circle(size=60).encode(
        x='bytes_sent',
        y='bytes_received',
        color=alt.condition(
            alt.datum.Alert == '🚨 Yes',
            alt.value('red'),  # Anomalies
            alt.value('blue')  # Normal
        ),
        tooltip=['bytes_sent', 'bytes_received', 'packets', 'Alert', 'malware_type', 'risk_level', 'timestamp']
    ).properties(
        title='Network Traffic: Bytes Sent vs. Bytes Received'
    )
    
    st.altair_chart(chart, use_container_width=True)

    # Packets Distribution (Binned)
    st.write("### 📦 Packets Distribution:")
    packets_chart = alt.Chart(data_with_anomalies).mark_bar().encode(
        x=alt.X('packets', bin=alt.Bin(maxbins=30)),
        y='count()',
        color=alt.condition(
            alt.datum.Alert == '🚨 Yes',
            alt.value('red'),
            alt.value('blue')
        )
    )
    
    st.altair_chart(packets_chart, use_container_width=True)
    
    # Malware Type Distribution
    st.write("### 🦠 Malware Type Distribution:")
    malware_chart = alt.Chart(data_with_anomalies[data_with_anomalies['malware_type'].notna()]).mark_bar().encode(
        x='malware_type',
        y='count()',
        color='malware_type'
    ).properties(
        title='Distribution of Malware Types in Anomalies'
    )
    
    st.altair_chart(malware_chart, use_container_width=True)
