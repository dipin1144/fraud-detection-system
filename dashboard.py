import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Real-Time Fraud Detection System")
st.markdown("Powered by XGBoost (AUC: 0.9566) | Trained on IEEE-CIS Dataset (590K transactions)")
st.divider()

# Sidebar — model info
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("AUC-ROC",   "0.9566", "+0.0298 vs baseline")
    st.metric("F1 Score",  "0.7020", "+0.3246 vs baseline")
    st.metric("Precision", "0.7803", "+0.4759 vs baseline")
    st.metric("Recall",    "0.6381")
    st.divider()
    st.caption("Model: XGBoost (Optuna tuned)")
    st.caption("Dataset: IEEE-CIS Fraud 590K txns")
    st.caption("Threshold: 0.5905 (optimal F1)")

# Main — input form
st.subheader("📝 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    product = st.selectbox("Product Code", ["W", "H", "C", "S", "R"])
    email = st.selectbox("Email Domain", ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "other"])

with col2:
    card1 = st.number_input("Card 1", min_value=0, value=9500)
    card4 = st.selectbox("Card Type", ["visa", "mastercard", "american express", "discover"])
    card6 = st.selectbox("Card Category", ["debit", "credit", "charge card"])

with col3:
    addr1 = st.number_input("Address Code", min_value=0, value=315)
    transaction_dt = st.number_input("Transaction Time (seconds)", min_value=0, value=86400)
    c1 = st.number_input("C1 (count feature)", min_value=0, value=1)

st.divider()

# Predict button
if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
    payload = {
        "TransactionAmt": amount,
        "ProductCD": product,
        "card1": card1,
        "card4": card4,
        "card6": card6,
        "addr1": addr1,
        "P_emaildomain": email,
        "TransactionDT": transaction_dt,
        "C1": c1
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        fraud_prob = result["fraud_probability"]
        is_fraud   = result["is_fraud"]
        risk_level = result["risk_level"]

        st.subheader("🎯 Prediction Result")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
        with col_b:
            st.metric("Decision", "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE")
        with col_c:
            color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            st.metric("Risk Level", f"{color} {risk_level}")

        # Progress bar
        st.progress(fraud_prob, text=f"Fraud Risk: {fraud_prob*100:.1f}%")

        if is_fraud:
            st.error("⚠️ This transaction has been flagged as potentially fraudulent!")
        else:
            st.success("✅ This transaction appears legitimate.")

    except Exception as e:
        st.error(f"API Error: {e}. Make sure the FastAPI server is running on port 8000.")
