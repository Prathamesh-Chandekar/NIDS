import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Network Security ML Dashboard",
    layout="wide",
    page_icon="🔒"
)

# Apply custom CSS
st.markdown("""
    <style>
    .stMetric div {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate sample network traffic data"""
    np.random.seed(42)  # For reproducible results
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
    
    data = {
        'timestamp': timestamps,
        'bytes_sent': np.random.randint(100, 10000, n_samples),
        'bytes_received': np.random.randint(100, 10000, n_samples),
        'packets': np.random.randint(1, 100, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'port': np.random.randint(1, 65535, n_samples)
    }
    return pd.DataFrame(data)

def detect_anomalies(data):
    """Perform anomaly detection on network traffic"""
    features = data[['bytes_sent', 'bytes_received', 'packets']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    detector = IsolationForest(contamination=0.1, random_state=42)
    anomalies = detector.fit_predict(scaled_features)
    return anomalies

def main():
    st.title("🔒 Network Security ML Dashboard")
    
    # Sidebar
    st.sidebar.title("Controls")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Traffic Overview", "Anomaly Detection", "Security Metrics"]
    )
    
    # Generate data
    data = generate_sample_data()
    
    if analysis_type == "Traffic Overview":
        st.header("Network Traffic Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Traffic", 
                     f"{data['bytes_sent'].sum() / 1e6:.2f} MB",
                     "Active")
        with col2:
            st.metric("Packets Processed", 
                     f"{data['packets'].sum():,}",
                     "Normal")
        with col3:
            st.metric("Active Protocols",
                     len(data['protocol'].unique()),
                     "Stable")
        
        # Traffic Volume Chart
        st.subheader("Network Traffic Volume")
        fig_traffic = px.line(
            data,
            x='timestamp',
            y=['bytes_sent', 'bytes_received'],
            title='Network Traffic Over Time'
        )
        st.plotly_chart(fig_traffic, use_container_width=True)
        
        # Protocol Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Protocol Distribution")
            protocol_dist = data['protocol'].value_counts()
            fig_protocol = px.pie(
                values=protocol_dist.values,
                names=protocol_dist.index
            )
            st.plotly_chart(fig_protocol)
            
        with col2:
            st.subheader("Port Usage")
            port_ranges = pd.cut(data['port'], 
                               bins=[0, 1024, 49151, 65535],
                               labels=['Well-Known', 'Registered', 'Dynamic'])
            port_dist = port_ranges.value_counts()
            fig_ports = px.bar(
                x=port_dist.index,
                y=port_dist.values,
                title='Port Range Distribution'
            )
            st.plotly_chart(fig_ports)
    
    elif analysis_type == "Anomaly Detection":
        st.header("Anomaly Detection")
        
        # Perform anomaly detection
        anomalies = detect_anomalies(data)
        data['anomaly'] = anomalies
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            anomaly_count = len(data[data['anomaly'] == -1])
            st.metric("Anomalies Detected", 
                     anomaly_count,
                     f"{(anomaly_count/len(data))*100:.1f}% of traffic")
        with col2:
            st.metric("Normal Traffic Patterns",
                     len(data[data['anomaly'] == 1]),
                     "Baseline")
        
        # Anomaly Visualization
        st.subheader("Traffic Anomaly Detection")
        fig_anomaly = px.scatter(
            data,
            x='bytes_sent',
            y='bytes_received',
            color='anomaly',
            title='Network Traffic Patterns',
            labels={'anomaly': 'Status'},
            color_discrete_map={1: 'blue', -1: 'red'}
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Show anomalous traffic
        if st.checkbox("Show Anomalous Traffic Details"):
            st.dataframe(
                data[data['anomaly'] == -1][['timestamp', 'bytes_sent', 
                                           'bytes_received', 'protocol', 'port']],
                hide_index=True
            )
    
    else:  # Security Metrics
        st.header("Security Metrics")
        
        # Calculate metrics
        total_traffic = data['bytes_sent'].sum() + data['bytes_received'].sum()
        avg_packet_size = total_traffic / data['packets'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Packet Size", 
                     f"{avg_packet_size:.2f} bytes",
                     "Normal")
        with col2:
            st.metric("Traffic Symmetry",
                     f"{(data['bytes_sent'].sum() / data['bytes_received'].sum()):.2f}",
                     "Balanced")
        with col3:
            st.metric("Unique Ports",
                     f"{data['port'].nunique():,}",
                     "Monitored")
        
        # Traffic pattern over time
        st.subheader("Traffic Pattern Analysis")
        hourly_traffic = data.resample('H', on='timestamp').sum()
        fig_hourly = px.line(
            hourly_traffic,
            y=['bytes_sent', 'bytes_received'],
            title='Hourly Traffic Pattern'
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

if __name__ == "__main__":
    main()
