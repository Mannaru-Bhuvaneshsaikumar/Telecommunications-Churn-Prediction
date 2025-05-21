import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model_path = "random_forest_model.pkl"  # Update with actual model path
scaler_path = "scaler.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn or not.")

# Input fields based on dataset features
account_length = st.number_input("Account Length", min_value=0, max_value=300, value=100)
voice_mail_plan = st.selectbox("Voice Mail Plan", [0, 1])
voice_mail_messages = st.number_input("Voice Mail Messages", min_value=0, max_value=50, value=10)
day_mins = st.number_input("Day Minutes", min_value=0.0, max_value=400.0, value=200.0)
evening_mins = st.number_input("Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
night_mins = st.number_input("Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
international_mins = st.number_input("International Minutes", min_value=0.0, max_value=50.0, value=10.0)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
international_plan = st.selectbox("International Plan", [0, 1])
day_calls = st.number_input("Day Calls", min_value=0, max_value=200, value=100)
day_charge = st.number_input("Day Charge", min_value=0.0, max_value=50.0, value=20.0)
evening_calls = st.number_input("Evening Calls", min_value=0, max_value=200, value=100)
evening_charge = st.number_input("Evening Charge", min_value=0.0, max_value=50.0, value=15.0)
night_calls = st.number_input("Night Calls", min_value=0, max_value=200, value=100)
night_charge = st.number_input("Night Charge", min_value=0.0, max_value=50.0, value=10.0)
international_calls = st.number_input("International Calls", min_value=0, max_value=20, value=3)
international_charge = st.number_input("International Charge", min_value=0.0, max_value=10.0, value=2.0)
total_charge = st.number_input("Total Charge", min_value=0.0, max_value=100.0, value=50.0)

# Add the missing feature (Assuming it's 'area_code')
area_code = st.number_input("Area Code", min_value=0, max_value=999, value=415)

# Ensure correct number of features
features = np.array([[account_length, voice_mail_plan, voice_mail_messages, day_mins, 
                       evening_mins, night_mins, international_mins, customer_service_calls, 
                       international_plan, day_calls, day_charge, evening_calls, evening_charge, 
                       night_calls, night_charge, international_calls, international_charge, total_charge, 
                       area_code]])  # Now includes 19 features

# Prediction button
if st.button("Predict"):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    
    # Special effect
    st.balloons()
    st.success(f"Prediction: {result}")
