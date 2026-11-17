import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_engine_condition_pipeline.joblib")
model = joblib.load(model_path)

st.title("Vehicle Engine condition Prediction App")
st.write("""
This application predicts the likelihood of vehicle engine maintenance required based on engine condition""")

st.write("Please enter the vehicle engine details below.")

EngineRPM = st.number_input(
    "Engine RPM",
    min_value=40,
    max_value=3000,
    value=100,
    step=1
)

LubOilPressure = st.number_input(
    "Lubricating Oil Pressure",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.000001,
    format="%.6f"
)

FuelPressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    max_value=30.0,
    value=10.0,
    step=0.000001,
    format="%.6f"
)

CoolantPressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.000001,
    format="%.6f"
)

LubOilTemp = st.number_input(
    "Lubricating Oil Temperature",
    min_value=50.0,
    max_value=100.0,
    value=70.0,
    step=0.00001,
    format="%.5f"
)

CoolantTemp = st.number_input(
    "Coolant Temperature",
    min_value=50.0,
    max_value=250.0,
    value=100.0,
    step=0.000001,
    format="%.6f"
)

input_data = pd.DataFrame([{
    "EngineRpm": EngineRPM,
    "LubOilPressure": LubOilPressure,
    "FuelPressure": FuelPressure,
    "CoolantPressure": CoolantPressure,
    "LubOilTemp": LubOilTemp,
    "CoolantTemp": CoolantTemp
}])

if st.button("Predict Engine Condition"):
    failure_probability = model.predict(input_data)[0]
    classification_threshold = 0.45
    prediction = int(
        failure_probability >= classification_threshold
    )
    st.write(f"Raw prediction: {prediction}")
    st.write(f"Failure probability: {failure_probability:.2%}")
    if prediction == 1:
      result = "Vehicle's Engine is likely to be faulty"
    else:
      result = "Vehicle's Engine is in good condition"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
