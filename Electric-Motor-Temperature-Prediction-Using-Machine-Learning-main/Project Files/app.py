import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# --- PAGE SETUP ---
st.set_page_config(page_title="Electric Motor Temperature Predictor", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_model_files():
    if not os.path.exists('model.save') or not os.path.exists('transform.save'):
        st.error("⚠️ Error: 'model.save' or 'transform.save' not found in the current folder.")
        st.stop()
    
    with open('model.save', 'rb') as f:
        model = pickle.load(f)
    with open('transform.save', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_files()

# --- HEADER ---
st.title("🌡️ Electric Motor Temperature Prediction")
st.markdown("Predicting **Rotor Temperature (PM)** based on sensor data for predictive maintenance.")
st.divider()

# --- INPUT SECTION ---
st.sidebar.header("📥 Sensor Measurements")

def get_user_inputs():
    # These 7 features must match the number of features your model was trained on
    ambient = st.sidebar.slider("Ambient Temperature", -10.0, 50.0, 25.0)
    coolant = st.sidebar.slider("Coolant Temperature", 0.0, 100.0, 40.0)
    u_d = st.sidebar.number_input("Voltage d-component (u_d)", value=0.0)
    u_q = st.sidebar.number_input("Voltage q-component (u_q)", value=0.0)
    motor_speed = st.sidebar.number_input("Motor Speed (RPM)", value=1500.0)
    i_d = st.sidebar.number_input("Current d-component (i_d)", value=0.0)
    i_q = st.sidebar.number_input("Current q-component (i_q)", value=0.0)
    
    # Create dictionary
    data = {
        'ambient': ambient,
        'coolant': coolant,
        'u_d': u_d,
        'u_q': u_q,
        'motor_speed': motor_speed,
        'i_d': i_d,
        'i_q': i_q
    }
    
    return pd.DataFrame(data, index=[0])

input_data = get_user_inputs()

# --- DISPLAY & PREDICTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Input Parameters")
    st.table(input_data)

with col2:
    st.subheader("Rotor Temperature Analysis")
    if st.button("Generate Prediction"):
        try:
            # THE FIX: Using .values converts the DataFrame to a raw Numpy array.
            # This skips the "Feature Names" validation error.
            raw_input = input_data.values 
            
            # Scale the data
            scaled_input = scaler.transform(raw_input)
            
            # Predict
            prediction = model.predict(scaled_input)[0]
            
            # Visual indicators
            st.metric(label="Predicted PM Temperature", value=f"{prediction:.2f} °C")
            
            if prediction > 65:
                st.error("🚨 HIGH TEMPERATURE: Maintenance Suggested.")
            elif prediction > 45:
                st.warning("⚠️ MODERATE: Monitor cooling systems.")
            else:
                st.success("✅ SAFE: Operating within normal range.")
                
        except ValueError as e:
            st.error(f"Shape Mismatch Error: {e}")
            st.info("This usually means your model expects more (or fewer) than 7 features. Check your Colab training script columns.")

st.divider()
st.caption("Developed using Scikit-Learn and Streamlit for Predictive Maintenance.")