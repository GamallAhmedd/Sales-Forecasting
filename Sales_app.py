
import streamlit as st
import joblib
import numpy as np

model = joblib.load('/content/ridge_model.pkl')

def main():
    st.title("Sales Prediction")
    st.sidebar.header("Input Features")
    year = st.sidebar.number_input("Year", min_value=2000, max_value=2025, value=2022)
    month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
    quarter = st.sidebar.number_input("Quarter", min_value=1, max_value=4, value=2)
    week = st.sidebar.number_input("Week", min_value=1, max_value=52, value=26)

    input_data = np.array([year, month, quarter, week]).reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == '__main__':
    main()
