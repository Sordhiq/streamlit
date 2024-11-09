import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        with open('logistics.pickle', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'model.pkl' exists in the same directory.")
        return None

def main():
    # Add title and description
    st.title("🌸 Iris Flower Prediction App")
    st.write("""
    This app predicts the type of Iris flower based on its measurements.
    Please input the following measurements in centimeters:
    """)
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    
    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    
    # Create a features array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Load the model
    model = load_model()
    
    if model is not None:
        # Add a prediction button
        if st.button("Predict"):
            # Make prediction
            prediction = model.predict(features)
            
            # Get prediction probability
            prediction_proba = model.predict_proba(features)
            
            # Get the class labels from the model if available
            if hasattr(model, 'classes_'):
                species_mapping = dict(enumerate(model.classes_))
            else:
                # Fallback mapping
                species_mapping = {
                    0: "Iris-setosa",
                    1: "Iris-versicolor",
                    2: "Iris-virginica"
                }
            
            # Display prediction
            predicted_species = species_mapping[prediction[0]] if isinstance(prediction[0], (int, np.integer)) else prediction[0]
            st.subheader("Prediction Results:")
            st.success(f"The Iris flower is predicted to be: **{predicted_species}**")
            
            # Display prediction probabilities
            st.subheader("Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Species': list(species_mapping.values()),
                'Probability': prediction_proba[0]
            })
            
            # Create a bar chart
            st.bar_chart(prob_df.set_index('Species'))
            
            # Display input data
            st.subheader("Input Features:")
            input_df = pd.DataFrame({
                'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
            })
            st.table(input_df)
    
    # Add information about the model
    with st.expander("ℹ️ About this app"):
        st.write("""
        This app uses a machine learning model trained on the famous Iris dataset.
        The model predicts the species of Iris flower based on four measurements:
        - Sepal Length
        - Sepal Width
        - Petal Length
        - Petal Width
        
        The possible species are:
        - Iris Setosa
        - Iris Versicolor
        - Iris Virginica
        """)

if __name__ == "__main__":
    main()
