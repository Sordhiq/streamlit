import pandas as pd
import numpy as np
import streamlit as st

st.title("Malaria Prediction App")
st.subtitle("This web application is aimed at using climatic variables in predicting malaria prevalences")

@st.cache_resource
def load_model():
  try:
    with open("damat.pkl", "rb") as file:
      model = pickle.load(file)
  except FileNotFoundError:
    st.error("You have attempted to load a wrong pickle file")
    return None

def pred():

  rainfall_rolling = number_input(value=17.0, min_value=0.0, max_value=263.0)
  rainfall_lag = number_input(value=50.0, min_value=0.0, max_value=400.0)
  cumulative_rainfall = number_input(value=4500.0, min_value=1000.0, max_value=8000.0)

  features = np.array([[rainfall_rolling, rainfall_lag, cumulative_rainfall]])
  prediction = model.predict(features)
  return prediction

models = pred()

if st.button("Predict"):
  st.success("The Predicted Malaria Cases is:", models)
  
st.expander("This model is proudly developed by Group 2 members of the CAN Data Science Fellowship")

if __main__ = __name__
  main()
