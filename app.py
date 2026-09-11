import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and threshold
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    threshold = joblib.load('best_threshold.pkl')
    return model, threshold

model, threshold = load_model()

# Title
st.title("🏭 Machine Failure Prediction")
st.write("Enter machine sensor data below to predict the likelihood of failure.")

# Sidebar for inputs
st.sidebar.header("Input Features")

# Define input widgets based on your features
# Note: Adjust these ranges based on your EDA min/max values
type_map = {'L': 0, 'M': 1, 'H': 2} # Assuming OneHotEncoder was used, check your preprocessor
product_type = st.sidebar.selectbox("Product Type", ['L', 'M', 'H'])

air_temp = st.sidebar.number_input("Air Temperature [K]", min_value=295.0, max_value=305.0, value=300.0)
process_temp = st.sidebar.number_input("Process Temperature [K]", min_value=305.0, max_value=315.0, value=310.0)
rotational_speed = st.sidebar.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500)
torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=80.0, value=40.0)
tool_wear = st.sidebar.number_input("Tool Wear [min]", min_value=0, max_value=250, value=100)

# Calculate derived features if your model expects them
# Check your 'features' list in the notebook. If you engineered features like 'Temperature Difference', calculate them here.
temp_diff = process_temp - air_temp
mechanical_load = torque * rotational_speed
torque_speed_ratio = torque / (rotational_speed + 1)

# Prepare the input data frame
# IMPORTANT: The column order MUST match the order used during training
input_data = pd.DataFrame({
    'Type': [product_type],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'Temperature Difference': [temp_diff],
    'Mechanical Load Proxy': [mechanical_load],
    'Torque Speed Ratio': [torque_speed_ratio]
})

# Prediction Button
if st.button("Predict Failure"):
    try:
        # Get probability
        probability = model.predict_proba(input_data)[0][1]
        
        # Apply threshold
        prediction = 1 if probability >= threshold else 0
        
        # Display Result
        if prediction == 1:
            st.error(f"⚠️ Warning: Machine Failure Predicted! (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Normal Operation. (Probability: {probability:.2f})")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Model trained on AI4I 2020 Predictive Maintenance Dataset")
