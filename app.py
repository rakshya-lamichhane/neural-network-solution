import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("best_pima_model.keras")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("Diabetes Risk Predictor (Pima Dataset)")
st.caption("Enter patient features below, then press **Predict**. Values should be realistic (non-zero for medical measures).")

# Sidebar controls
st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider("Decision threshold (default 0.50)", 0.05, 0.95, 0.50, 0.01)
show_scaled = st.sidebar.checkbox("Show scaled features used by the model", value=False)

feature_names = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

# Reasonable input widgets with ranges & units
col1, col2, col3, col4 = st.columns(4)
with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
with col2:
    Glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=120.0, step=1.0)
with col3:
    BloodPressure = st.number_input("BloodPressure (mm Hg)", min_value=40.0, max_value=140.0, value=70.0, step=1.0)
with col4:
    SkinThickness = st.number_input("SkinThickness (mm)", min_value=5.0, max_value=80.0, value=25.0, step=1.0)

col5, col6, col7, col8 = st.columns(4)
with col5:
    Insulin = st.number_input("Insulin (µU/mL)", min_value=10.0, max_value=900.0, value=100.0, step=1.0)
with col6:
    BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=30.0, step=0.1, format="%.1f")
with col7:
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.05, max_value=3.0, value=0.5, step=0.01, format="%.2f")
with col8:
    Age = st.number_input("Age (years)", min_value=21, max_value=90, value=45, step=1)

# Sample patient autofill
def set_sample():
    st.session_state.update({
        "Pregnancies":2,"Glucose":130.0,"BloodPressure":72.0,"SkinThickness":25.0,
        "Insulin":100.0,"BMI":30.1,"DiabetesPedigreeFunction":0.5,"Age":45
    })

st.sidebar.button("Use sample patient", on_click=set_sample)

# Predict
if st.button("Predict"):
    X = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=feature_names).astype(float)

    # Use the same scaler from training
    Xs = scaler.transform(X)
    prob = float(model.predict(Xs, verbose=0)[0][0])
    label = "Diabetes" if prob >= threshold else "No Diabetes"

    st.subheader("Result")
    st.metric("Probability of Diabetes", f"{prob:.2%}")
    st.write("Prediction at threshold", threshold, "→ **", label, "**")

    if show_scaled:
        st.write("Scaled features used by the model:")
        st.dataframe(pd.DataFrame(Xs, columns=feature_names).round(3))

st.markdown("---")
st.caption("Note: This app is for educational purposes and not a medical device.")
