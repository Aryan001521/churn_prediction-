import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model (make sure xgboost installed and pkl is in same folder)
model = joblib.load("xgboost_churn_model.pkl")

st.title("💡 Customer Churn Prediction (XGBoost)")

# ---- Sidebar: explanation and mappings ----
st.sidebar.header("Info & Encodings")

st.sidebar.markdown("""
**What do predictions mean?**  
- **0 = Stay (No churn)**  
- **1 = Churn (Will leave)**

**Probability**: model predicts probability of churn (0 to 1).  
If probability ≥ 0.5 → model predicts *Churn* by default.
""")

st.sidebar.markdown("**Action guide (example):**\n- < 0.3 → Low risk\n- 0.3–0.6 → Medium risk (send offer)\n- ≥ 0.6 → High risk (call / retention offer)")

st.sidebar.markdown("**Feature encodings (adjust if your dataset differs):**")
st.sidebar.markdown("""
- gender: 0 = Male, 1 = Female  
- SeniorCitizen: 0 = No, 1 = Yes  
- Contract: 0 = Month-to-month, 1 = One year, 2 = Two year  
- PaperlessBilling: 0 = No, 1 = Yes  
- PaymentMethod: 0..3 (map to your dataset)  
""")

# ---- Input form with friendly labels ----
with st.form("input_form"):
    st.subheader("Enter customer details")
    gender_label = st.radio("Gender", ("Male", "Female"))
    gender = 0 if gender_label == "Male" else 1

    senior = st.selectbox("Senior Citizen?", ("No", "Yes"))
    SeniorCitizen = 0 if senior == "No" else 1

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    contract_label = st.selectbox("Contract type", ("Month-to-month", "One year", "Two year"))
    Contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract_label]

    pb_label = st.radio("Paperless Billing?", ("No", "Yes"))
    PaperlessBilling = 0 if pb_label == "No" else 1

    # Payment method choices - show readable labels but convert to numeric mapping
    pm_label = st.selectbox("Payment Method",
                            ("Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"))
    PaymentMethod_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (auto)": 2, "Credit card (auto)": 3}
    PaymentMethod = PaymentMethod_map[pm_label]

    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.1)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2500.0, step=0.1)
    TotalServiceused = st.slider("Total Services Used", 0, 15, 5)
    FamilyMembers = st.slider("Family Members", 0, 10, 1)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    # Build a DataFrame exactly like model expects
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'TotalServiceused': TotalServiceused,
        'FamilyMembers': FamilyMembers
    }])

    # If you used scaling/encoders during training, apply same transforms here!
    # Example: if you used a scaler saved as scaler.pkl, load & transform:
    # scaler = joblib.load("scaler.pkl")
    # input_df[scal_cols] = scaler.transform(input_df[scal_cols])

    # Predict
    proba = model.predict_proba(input_df)[0][1]  # probability of churn
    pred = int(proba >= 0.5)  # default threshold 0.5

    # Human-friendly display
    label_text = "CHURN (will leave)" if pred == 1 else "STAY (will NOT leave)"
    st.markdown(f"### Prediction: **{label_text}**")
    st.info(f"Churn probability: **{proba:.2f}** (threshold 0.50)")

    # Recommended action based on probability
    if proba < 0.3:
        st.success("Low risk — no immediate action required.")
    elif proba < 0.6:
        st.warning("Medium risk — consider offering a promotion or engagement email.")
    else:
        st.error("High risk — recommend calling the customer and offering a retention plan.")

    # Show the numeric mapping for clarity
    st.write("#### Mappings used (for clarity):")
    st.write(pd.DataFrame([
        ["gender", "0 = Male, 1 = Female"],
        ["Contract", "0 = Month-to-month, 1 = One year, 2 = Two year"],
        ["PaymentMethod", "0 = Electronic check, 1 = Mailed check, 2 = Bank transfer (auto), 3 = Credit card (auto)"]
    ], columns=["Feature", "Mapping"]))
