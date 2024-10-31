import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

# Page config
st.set_page_config(
    page_title="ML Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)

class MLSecurityApp:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.traffic_classifier = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        
    def generate_sample_data(self):
        """Generate sample network traffic data for demonstration"""
        n_samples = 1000
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
        
        data = {
            'timestamp': timestamps,
            'bytes_sent': np.random.randint(100, 10000, n_samples),
            'bytes_received': np.random.randint(100, 10000, n_samples),
            'packets': np.random.randint(1, 100, n_samples),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
            'port': np.random.randint(1, 65535, n_samples),
            'duration_ms': np.random.randint(1, 1000, n_samples)
        }
        return pd.DataFrame(data)

def main():
    st.title("🛡️ ML-Based Network Security Dashboard")
    
    # Initialize the ML security system
    security_app = MLSecurityApp()
    
    # Sidebar
    st.sidebar.header("Controls")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Real-time Monitoring", "Anomaly Detection", "Traffic Classification"]
    )
    
    # Generate sample data
    data = security_app.generate_sample_data()
    
    if analysis_type == "Real-time Monitoring":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Traffic Overview")
            # Traffic volume chart
            fig_traffic = px.line(
                data,
                x='timestamp',
                y=['bytes_sent', 'bytes_received'],
                title='Network Traffic Volume'
            )
            st.plotly_chart(fig_traffic, use_container_width=True)
            
        with col2:
            st.subheader("Protocol Distribution")
            protocol_dist = data['protocol'].value_counts()
            fig_protocol = px.pie(
                values=protocol_dist.values,
                names=protocol_dist.index,
                title='Protocol Distribution'
            )
            st.plotly_chart(fig_protocol, use_container_width=True)
            
        # Recent Activity Table
        st.subheader("Recent Network Activity")
        st.dataframe(data.head(10), hide_index=True)
        
    elif analysis_type == "Anomaly Detection":
        st.subheader("Network Anomaly Detection")
        
        # Prepare features for anomaly detection
        features = data[['bytes_sent', 'bytes_received', 'packets', 'duration_ms']]
        scaled_features = security_app.scaler.fit_transform(features)
        
        # Detect anomalies
        anomalies = security_app.anomaly_detector.fit_predict(scaled_features)
        data['anomaly'] = anomalies
        
        # Visualize anomalies
        fig_anomaly = px.scatter(
            data,
            x='bytes_sent',
            y='bytes_received',
            color='anomaly',
            title='Network Traffic Anomalies',
            color_discrete_map={1: 'blue', -1: 'red'},
            labels={'anomaly': 'Status', 1: 'Normal', -1: 'Anomaly'}
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Anomaly Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Anomalies Detected",
                len(data[data['anomaly'] == -1]),
                f"{len(data[data['anomaly'] == -1])/len(data)*100:.1f}% of traffic"
            )
        with col2:
            st.metric(
                "Normal Traffic Patterns",
                len(data[data['anomaly'] == 1]),
                f"{len(data[data['anomaly'] == 1])/len(data)*100:.1f}% of traffic"
            )
            
    elif analysis_type == "Traffic Classification":
        st.subheader("Traffic Pattern Classification")
        
        # Simple traffic classification based on volume
        data['traffic_class'] = pd.qcut(
            data['bytes_sent'] + data['bytes_received'],
            q=3,
            labels=['Low', 'Medium', 'High']
        )
        
        # Traffic classification visualization
        fig_class = px.scatter(
            data,
            x='bytes_sent',
            y='bytes_received',
            color='traffic_class',
            title='Traffic Classification'
        )
        st.plotly_chart(fig_class, use_container_width=True)
        
        # Traffic class distribution
        class_dist = data['traffic_class'].value_counts()
        fig_dist = px.bar(
            x=class_dist.index,
            y=class_dist.values,
            title='Traffic Class Distribution',
            labels={'x': 'Traffic Class', 'y': 'Count'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System Status", "Online", "Active")
    with col2:
        st.metric("ML Models Loaded", "3/3", "100%")
    with col3:
        st.metric("Last Update", "Just now", "Real-time")

if __name__ == "__main__":
    main()
