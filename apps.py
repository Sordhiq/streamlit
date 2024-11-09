import numpy as np
import pickle
import pandas as pd
import streamlit as st 

with open("logistics.pkl", "rb") as file:
    classifier = pickle.load(file)

def welcome():
    return "Welcome All"

def predict_iris(Sepal_length, Sepal_width, Petal_length, Petal_width):
    
    prediction = classifier.predict([[Sepal_length, Sepal_width, Petal_length, Petal_width]])
    return prediction

def main():
    st.title("AIR_lthy")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Air Quality Index Prediction App </h2>
    </div>
    """
        
    """Proudly, The Outliers!"""


    st.markdown(html_temp,unsafe_allow_html=True)
    Sepal_length = st.text_input("Sepal Lenght")
    Sepal_width = st.text_input("Sepal Width")
    Petal_length = st.text_input("Petal Lenght")
    Petal_width = st.text_input("Petal Width")
    result=""
    if st.button("Predict"):
        result = predict_iris(Sepal_length, Sepal_width, Petal_length, Petal_width)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Many thanks")

if __name__=='__main__':
    main()
