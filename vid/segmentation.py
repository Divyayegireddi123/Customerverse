import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Customer Segmentation App")
st.subheader("Machine Learning Based Customer Classification System")

st.write("Enter Customer details to predict the segment.")

st.info(
    "This application predicts customer segments using "
    "K-Means Clustering based on customer behavior and spending patterns."
)

# User Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

income = st.number_input(
    "Income",
    min_value=0,
    max_value=20000,
    value=5000
)

total_spending = st.number_input(
    "Total Spending (sum of purchases)",
    min_value=0,
    max_value=5000,
    value=1000
)

num_web_purchases = st.number_input(
    "Number of Web Purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_store_purchases = st.number_input(
    "Number of Store Purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_web_visits = st.number_input(
    "Number of Web Visits per Month",
    min_value=0,
    max_value=50,
    value=3
)

recency = st.number_input(
    "Recency (days since last purchase)",
    min_value=0,
    max_value=365,
    value=30
)

# Create DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict Button
if st.button("Predict Segment"):

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Cluster Names
    cluster_names = {
        0: "High Value Customers",
        1: "Regular Customers",
        2: "Low Spending Customers",
        3: "Frequent Buyers",
        4: "Occasional Customers",
        5: "Premium Customers"
    }

    # Show Result
    st.success(
        f"Predicted Segment: Cluster {cluster} - {cluster_names.get(cluster)}"
    )