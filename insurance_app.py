import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
 
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰", layout="centered")
 
st.title("Medical Insurance Cost Prediction")
st.write("Enter the patient's details below to estimate their annual insurance cost.")
 
# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = "model.pkl"
 
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model
 
model = load_model()
 
if model is None:
    st.error(
        f"Could not find '{MODEL_PATH}'. Please export your trained model "
        f"from your training notebook/script and place it next to this app. "
        f"See the bottom of insurance_app.py for a saving example."
    )
    st.stop()
 
# -----------------------------
# Input fields
# -----------------------------
st.subheader("Patient Information")
 
col1, col2 = st.columns(2)
 
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
 
with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoking Status", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
 
# -----------------------------
# Prepare input for the model
# -----------------------------
# IMPORTANT: This encoding MUST match exactly how you encoded these columns
# when you trained your model in an earlier phase. Adjust if your training
# pipeline used different encoding (e.g. OneHotEncoder, get_dummies, etc.)
 
def preprocess_input(age, bmi, children, sex, smoker, region):

    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,

        "sex_female": 1 if sex == "female" else 0,
        "sex_male": 1 if sex == "male" else 0,

        "smoker_no": 1 if smoker == "no" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,

        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }])

    return input_df
# -----------------------------
# Predict button
# -----------------------------
st.write("")
if st.button("🔮 Predict Insurance Cost", use_container_width=True):
    input_data = preprocess_input(age, bmi, children, sex, smoker, region)
 
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### Estimated Insurance Cost: ${prediction:,.2f}")
 
        with st.expander("See input summary"):
            st.write(input_data)
 
    except Exception as e:
        st.error(
            "Prediction failed. This usually means the column names/order "
            "your model expects don't match what this app is sending. "
            f"Error: {e}"
        )
        st.info(
            "Fix: open your training script, check X_train.columns (or the "
            "feature list used in model.fit), and update preprocess_input() "
            "in this file to match exactly."
        )
 
st.divider()
