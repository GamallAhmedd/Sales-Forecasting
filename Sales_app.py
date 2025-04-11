import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('ridge_model.pkl')

# Streamlit App
def main():
    st.title("Sales Prediction")

    # Create a container for the inputs
    with st.container():
        # Collect inputs from the user
        year = st.number_input("Year", min_value=2000, max_value=2025, value=2022)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        quarter = st.number_input("Quarter", min_value=1, max_value=4, value=2)
        week = st.number_input("Week", min_value=1, max_value=52, value=26)

        # Prepare input data for prediction
        input_data = np.array([year, month, quarter, week]).reshape(1, -1)

        # Add space for styling
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Predict"):
            # Make prediction
            prediction = model.predict(input_data)
            st.write(f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == '__main__':
    main()
