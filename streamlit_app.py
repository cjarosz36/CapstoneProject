import streamlit as st
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("spending_model.pkl")

model = load_model()

# Streamlit UI
st.title("Steam Spending Predictor")

st.sidebar.header("User Input Features")
games_owned = st.sidebar.number_input("Number of Games Owned", min_value=0, max_value=1000, value=10)

if st.sidebar.button("Predict Spending"):
    features = np.array([[games_owned]])  # Adjust based on your model
    prediction = model.predict(features)
    st.subheader(f"Estimated Spending: ${prediction[0]:,.2f}")

