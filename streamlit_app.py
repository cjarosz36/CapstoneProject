import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("user_spending.csv")

# Load trained model
model = joblib.load("spending_model.pkl")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["EDA", "Modeling"])

if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    st.write("### User Spending Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["total_spent_usd"], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Total Spent (USD)")
    ax.set_ylabel("Number of Users")
    st.pyplot(fig)

    st.write("### Top 10 Spending Users")
    top_users = df.sort_values(by="total_spent_usd", ascending=False).head(10)
    st.bar_chart(top_users.set_index("playerid")["total_spent_usd"])

elif page == "Modeling":
    st.title("User Spending Prediction")

    st.write("Enter user details to predict spending:")
    num_games_owned = st.number_input("Number of Games Owned", min_value=0, step=1)
    
    # Make prediction
    prediction = model.predict([[num_games_owned]])[0]
    st.write(f"### Predicted Spending: **${prediction:.2f}**")


