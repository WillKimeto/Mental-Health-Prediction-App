import streamlit as st
import xgboost as xgb
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model("xgboost_model.json")

# Custom CSS with soft pink background and modern button styling
st.markdown(
    """
    <style>
    body {
        background-color: #ffccd5; /* Soft pink background */
    }
    .main-content {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #ffffff; /* White content area */
    }
    .title {
        color: #e74c3c;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .description {
        color: #7f8c8d;
        font-size: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #c0392b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<p class="title">Mental Health Treatment Prediction App</p>', unsafe_allow_html=True)

# Introduction Section
st.markdown(
    """
    <p class="description">
    Welcome to the Mental Health Treatment Prediction App!  
    This web application is part of my final-year university project, where I’ve used machine learning to explore how factors like occupation, lifestyle, and access to care can help identify potential mental health risks.

The goal of this project is to use data to provide early insights, so that individuals and communities can take proactive steps toward better mental well-being.

How it works:

You’ll answer a few questions about your background, work life, and lifestyle habits.

The model will analyze these inputs, compare them with patterns from the dataset, and provide an estimated risk profile.

Important:
This tool is not a medical diagnosis. It’s designed for educational purposes and to raise awareness, not to replace professional advice. If you have concerns about your mental health, please reach out to a qualified professional.

When you’re ready, scroll down and enter your details to get started  
    </p>
    """,
    unsafe_allow_html=True
)



# Get Started Button
if st.button("Get Started"):
    st.session_state.show_inputs = True

# Input Fields Section (hidden until "Get Started" clicked)
if st.session_state.get("show_inputs", False):

    st.subheader("Enter Patient Details")

    # Define feature inputs
    family_history = st.selectbox("Family History of Mental Health Issues", ["Yes", "No"])
    days_indoors = st.selectbox("Days Indoors", ["1-14 days", "15-30 days", "More than 2 months"])
    growing_stress = st.selectbox("Growing Stress", ["Yes", "No", "Maybe"])

    # Label Encoding
    le_family_history = LabelEncoder()
    le_days_indoors = LabelEncoder()
    le_growing_stress = LabelEncoder()

    le_family_history.classes_ = np.array(["No", "Yes"])
    le_days_indoors.classes_ = np.array(["1-14 days", "15-30 days", "More than 2 months"])
    le_growing_stress.classes_ = np.array(["Maybe", "No", "Yes"])

    input_data = pd.DataFrame({
        "family_history": [le_family_history.transform([family_history])[0]],
        "Days_Indoors": [le_days_indoors.transform([days_indoors])[0]],
        "Growing_Stress": [le_growing_stress.transform([growing_stress])[0]]
    })

    # Make prediction
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    probability = prediction[0]

    st.write(f"Predicted Probability of Seeking Treatment: {probability:.2%}")
    if probability > 0.5:
        st.success("Prediction: Likely to seek treatment (Yes)")
    else:
        st.warning("Prediction: Unlikely to seek treatment (No)")

    # SHAP Explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP visualization could not be displayed: {e}")
