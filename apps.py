import numpy as np  
import pickle
import pandas as pd
import streamlit as st 

@st.cache_resource
def load_model():
    try:
        with open('logistics.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("The file you have attempted to load does not exist in the file directory.\
                 Kindly recheck please.")
        return None
        
classifier = load_model()

def welcome():
    return "Welcome All"

def predict_iris(Sepal_length, Sepal_width, Petal_length, Petal_width):

    features = np.array([[Sepal_length, Sepal_width, Petal_length, Petal_width]])
    prediction = classifier.predict(features)
    return prediction

def main():
    st.title("Predicting Iris Flowers")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Air Quality Index Prediction App </h2>
    </div>
    """
        
    """Proudly, The Outliers!"""


    st.markdown(html_temp,unsafe_allow_html=True)
    Sepal_length = st.number_input("Sepal Lenght", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    Sepal_width = st.number_input("Sepal Width")
    Petal_length = st.number_input("Petal Lenght", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    Petal_width = st.number_input("Petal Width")
    result=""
    if st.button("Predict"):
        result = predict_iris(Sepal_length, Sepal_width, Petal_length, Petal_width)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Many thanks")

if __name__=='__main__':
    main()


