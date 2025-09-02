import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load the model and scaler
model = tf.keras.models.load_model("best_pima_model.keras")
scaler = joblib.load("scaler.joblib")

st.title("Diabetes Risk Predictor (Pima Dataset)")

# Define input features
feature_names = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

# Input fields
inputs = []
cols = st.columns(4)  # split into 4 columns for cleaner layout
for i, f in enumerate(feature_names):
    with cols[i % 4]:
        val = st.number_input(f, step=0.1)
        inputs.append(val)

# Prediction button
if st.button("Predict"):
    X = np.array([inputs]).astype(float)
    Xs = scaler.transform(pd.DataFrame(X, columns=feature_names))
    prob = float(model.predict(Xs, verbose=0)[0][0])
    st.metric("Diabetes Probability", f"{prob:.2%}")
    st.write("Prediction:", "Diabetes" if prob >= 0.5 else "No Diabetes")
