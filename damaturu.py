import pandas as pd
import numpy as np
import pickle
import streamlit as st

@st.cache_resource
def load_model():
  try:
    with open("damat.pkl", "rb") as file:
      mod = pickle.load(file)
    return mod
    
  except FileNotFoundError:
    st.error("You have attempted to load a wrong pickle file")
    return None

model = load_model()

def predi(rainfall_rolling, rainfall_lag, cumulative_rainfall):
  features = np.array([[rainfall_rolling, rainfall_lag, cumulative_rainfall]])
  prediction = model.predict(features)
  return prediction

  
def main():

  st.title("Malaria Prediction App")
  st.subheader("This web application is aimed at using climatic variables in predicting malaria prevalences")
    
  rainfall_rolling = st.number_input("Rainfall Rolling Average", value=17.0, min_value=0.0, max_value=263.0)
  rainfall_lag = st.number_input("Ranfall Lag 3", value=50.0, min_value=0.0, max_value=400.0)
  cumulative_rainfall = st.number_input("Cumulative Rainfall", value=4500.0, min_value=1000.0, max_value=8000.0)
  
  if st.button("Predict"):
    predictions = predi(rainfall_rolling, rainfall_lag, cumulative_rainfall)
    st.success(f"The Predicted Malaria Cases is: {np.round(predictions, 1)}")
    
  st.expander("This model is proudly developed by Group 2 members of the CAN Data Science Fellowship")

if __name__=='__main__':
    main()
