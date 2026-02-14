import pandas as pd
import joblib
import streamlit as st

# Load trained pipeline model
model = joblib.load("fraud_detection_model.pkl")

# App title
st.title("Fraud Detection Prediction App")

st.markdown("Please enter the transaction details and click the Predict button.")
st.divider()

# User inputs
transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input(
    "Old Balance (Sender)", min_value=0.0, value=10000.0
)

newbalanceOrig = st.number_input(
    "New Balance (Sender)", min_value=0.0, value=9000.0
)

oldbalanceDest = st.number_input(
    "Old Balance (Receiver)", min_value=0.0, value=0.0
)

newbalanceDest = st.number_input(
    "New Balance (Receiver)", min_value=0.0, value=0.0
)

# Predict button
if st.button("Predict"):

    # Prepare input data (column names must match training data)
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction: {int(prediction)}")

    if prediction == 1:
        st.error("⚠️ This transaction is likely FRAUD")
    else:
        st.success("✅ This transaction is NOT a fraud")