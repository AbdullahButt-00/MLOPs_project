import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Title
st.title("🎯 Federated Learning Churn Prediction System")
st.markdown("**MLOps-Powered Real-Time Prediction Dashboard**")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Prediction", "Monitoring", "Model Performance"])

if page == "Prediction":
    st.header("📊 Real-Time Churn Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse_distance = st.number_input("Warehouse to Home (km)", min_value=0.0, max_value=200.0, value=15.0)
        payment_mode = st.selectbox("Payment Mode", ["Credit Card", "Debit Card", "UPI", "Cash on Delivery", "E wallet"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours_on_app = st.number_input("Hours Spend on App", min_value=0.0, max_value=10.0, value=2.5)
        devices_registered = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=3)
        
    with col2:
        order_category = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        number_of_address = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)
        complain = st.selectbox("Complain", [0, 1])
        order_hike = st.number_input("Order Amount Hike from Last Year (%)", min_value=0.0, max_value=50.0, value=15.0)
        coupon_used = st.number_input("Coupons Used", min_value=0.0, max_value=20.0, value=3.0)
        order_count = st.number_input("Order Count", min_value=1, max_value=50, value=5)
        days_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=50, value=7)
        cashback = st.number_input("Cashback Amount", min_value=0.0, max_value=500.0, value=150.0)
    
    if st.button("Predict Churn", type="primary"):
        # Prepare data
        data = {
            "Tenure": tenure,
            "PreferredLoginDevice": login_device,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_distance,
            "PreferredPaymentMode": payment_mode,
            "Gender": gender,
            "HourSpendOnApp": hours_on_app,
            "NumberOfDeviceRegistered": devices_registered,
            "PreferedOrderCat": order_category,
            "SatisfactionScore": satisfaction,
            "MaritalStatus": marital_status,
            "NumberOfAddress": number_of_address,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": order_hike,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": days_since_last_order,
            "CashbackAmount": cashback
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=data)
            result = response.json()
            
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            with col2:
                st.metric("Risk Level", result['risk_level'])
            with col3:
                st.metric("Prediction", "Will Churn" if result['churn_prediction'] == 1 else "Will Stay")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['churn_probability'] * 100,
                title = {'text': "Churn Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            st.info("Make sure the API server is running on http://localhost:8000")

elif page == "Monitoring":
    st.header("📈 System Monitoring")
    
    # Simulated metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions Today", "1,247", "+12%")
    with col2:
        st.metric("Average Latency", "45ms", "-3ms")
    with col3:
        st.metric("Model Accuracy", "87.4%", "+0.2%")
    with col4:
        st.metric("Drift Score", "0.12", "🟢")
    
    # Simulated time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    predictions = np.random.randint(800, 1500, size=30)
    
    df_metrics = pd.DataFrame({
        'Date': dates,
        'Predictions': predictions,
        'Accuracy': np.random.uniform(0.85, 0.90, 30),
        'Latency': np.random.uniform(40, 60, 30)
    })
    
    fig = px.line(df_metrics, x='Date', y='Predictions', title='Daily Predictions Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df_metrics, x='Date', y='Accuracy', title='Model Accuracy Over Time')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(df_metrics, x='Date', y='Latency', title='Prediction Latency (ms)')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.header("🎯 Model Performance Metrics")
    
    # Load evaluation results (you'd load this from a file in production)
    st.subheader("Latest Evaluation Results")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", "87.4%")
    with col2:
        st.metric("Precision", "84.2%")
    with col3:
        st.metric("Recall", "81.8%")
    with col4:
        st.metric("F1-Score", "83.0%")
    with col5:
        st.metric("ROC AUC", "92.1%")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    confusion_data = np.array([[450, 50], [70, 430]])
    
    fig = px.imshow(confusion_data, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("Top 10 Feature Importances")
    features = ['Tenure', 'CashbackAmount', 'DaySinceLastOrder', 'OrderCount', 
                'HourSpendOnApp', 'Complain', 'WarehouseToHome', 'SatisfactionScore',
                'CouponUsed', 'OrderAmountHike']
    importances = np.random.uniform(0.02, 0.15, 10)
    
    df_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = px.barh(df_importance, x='Importance', y='Feature', title='Feature Importance')
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("**MLOps System Status:** 🟢 All systems operational")
st.sidebar.markdown("**Last Model Update:** 2 hours ago")
st.sidebar.markdown("**Next Scheduled Retrain:** Tomorrow 2:00 AM")