import streamlit as st
import pandas as pd
import pickle  # For loading the trained model
from sklearn.preprocessing import LabelEncoder

# ✅ Move set_page_config to the top
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

# Load the custom CSS file safely
def load_css():
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Default styling applied.")

# Call CSS function
load_css()

# ✅ Load the trained model safely
model_path = "credit_risk_model.pkl"
try:
    with open('credit-risk.pkl', "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Please ensure `credit_risk_model.pkl` exists.")
    model = None

# ✅ Function to preprocess user input (No fitting, only transformation)
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Correct encoding (Ensure encoders were trained with the model)
    label_encoders = {
        "grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
        "home_ownership": {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3},
        "verification_status": {"Verified": 0, "Not Verified": 1, "Source Verified": 2},
        "title": {"Debt Consolidation": 0, "Credit Card Refinancing": 1, "Home Improvement": 2, "Other": 3},
        "initial_list_status": {"w": 0, "f": 1}
    }

    # Apply encoding
    for col, mapping in label_encoders.items():
        df[col] = df[col].map(mapping)

    return df

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Predict Credit Risk"])

# ✅ **PAGE 1: Introduction**
if page == "Introduction":
    st.title("Credit Risk Prediction")
    st.write("""
    Welcome to the **Credit Risk Prediction** tool.  
    This application helps assess the likelihood of loan repayment based on financial and categorical attributes.

    ### How It Works:
    - The model predicts whether a loan falls under **low risk (0) or high risk (1)**.
    - It considers factors like **loan amount, interest rate, credit history, and financial status**.

    ### Instructions:
    1. Navigate to the **Predict Credit Risk** page.
    2. Enter the required details.
    3. Click **Submit** to see the result.
    """)

# ✅ **PAGE 2: Prediction**
elif page == "Predict Credit Risk":
    st.title("Loan Risk Assessment")
    st.write("Enter the details below to assess the credit risk.")

    # ✅ User Input Form
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=5000000, step=500)
    term = st.selectbox("Loan Term (Months)", [24, 36, 60])
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, step=0.1)
    grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=5000000, step=1000)
    verification_status = st.selectbox("Verification Status", ["Verified", "Not Verified", "Source Verified"])
    title = st.selectbox("Loan Purpose", ["Debt Consolidation", "Credit Card Refinancing", "Home Improvement", "Other"])
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 40.0, step=0.1)
    open_acc = st.number_input("Open Accounts", min_value=1, max_value=50, step=1)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=5000000, step=1000)
    revol_util = st.slider("Revolving Credit Utilization (%)", 0.0, 100.0, step=0.1)
    total_acc = st.number_input("Total Accounts", min_value=1, max_value=100, step=1)
    initial_list_status = st.selectbox("Initial List Status", ["w", "f"])
    total_pymnt = st.number_input("Total Payment Made ($)", min_value=0, max_value=5000000, step=1000)
    total_rec_int = st.number_input("Total Interest Received ($)", min_value=0, max_value=5000000, step=100)
    tot_cur_bal = st.number_input("Total Current Balance ($)", min_value=0, max_value=10000000, step=1000)

    # ✅ Submit button
    if st.button("Submit"):
        # Prepare input data
        input_data = {
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "grade": grade,
            "home_ownership": home_ownership, "annual_inc": annual_inc, "verification_status": verification_status,
            "title": title, "dti": dti, "open_acc": open_acc, "revol_bal": revol_bal, "revol_util": revol_util,
            "total_acc": total_acc, "initial_list_status": initial_list_status, "total_pymnt": total_pymnt,
            "total_rec_int": total_rec_int, "tot_cur_bal": tot_cur_bal
        }

        # Preprocess input
        processed_data = preprocess_input(input_data)

        if model:
            probabilities = model.predict_proba(processed_data)[0]  # Get probability scores
            high_risk_prob = probabilities[1]  # Assuming 1 represents high risk

            # Define risk categories based on probability
            if high_risk_prob >= 0.8:
                risk_level = "Very High Risk"
            elif high_risk_prob >= 0.6:
                risk_level = "High Risk"
            elif high_risk_prob >= 0.4:
                risk_level = "Moderate Risk"
            elif high_risk_prob >= 0.2:
                risk_level = "Low Risk"
            else:
                risk_level = "Very Low Risk"

            st.success(f"Prediction: **{risk_level}** ({high_risk_prob:.2%} probability)")
        else:
            st.error("Model not found. Please ensure the model file is available.")

