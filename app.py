import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Function to load the saved model, label encoder, and scaler
def load_randomforest():
    model = joblib.load('random_forest_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')  
    scaler = joblib.load('scaler.pkl')
    return model, label_encoder, scaler

# Function for model selection buttons
def selected_model():
    col1 = st.columns(1)[0]

    with col1:
        RF = st.button('Random Forest')
        if RF:
            st.session_state['selected_model'] = 'random_forest'
            st.session_state['model'], st.session_state['label_encoder'], st.session_state['scaler'] = load_randomforest()
            st.success("Random Forest model loaded successfully!")

# Title of the app
st.title("Bank Account Prediction App")

# Model selection
selected_model()

# Function to encode categorical features using LabelEncoder
def encode_features(input_data: pd.DataFrame, categorical_features: list):
    encoded_data = input_data.copy()
    
    # Initialize LabelEncoder if not in session state
    if 'label_encoder' in st.session_state:
        encoder = st.session_state['label_encoder']
    else:
        encoder = LabelEncoder() 
    
    # Apply LabelEncoder to each categorical column
    for feature in categorical_features:
        encoded_data[feature] = encoder.fit_transform(input_data[feature])
        
    return encoded_data

# Function to make predictions using the selected model
def make_prediction(features: pd.DataFrame):
    if 'model' not in st.session_state or 'scaler' not in st.session_state or 'label_encoder' not in st.session_state:
        st.error("Please select a model first!")
        return None, None

    # Define categorical and numerical features
    categorical_features = ['country', 'location_type', 'cellphone_access', 
                            'relationship_with_head', 'education_level', 'job_type', 'age_group']
    numerical_features = ['household_size', 'year']

    # Encode categorical columns
    encoded_features = encode_features(features, categorical_features)

    # Scale numerical columns
    scaled_features = st.session_state['scaler'].transform(encoded_features[numerical_features])
    encoded_features[numerical_features] = scaled_features

    # Make prediction using the selected model
    model = st.session_state['model']
    prediction = model.predict(encoded_features)

    # Handle the probability prediction
    try:
        probability = model.predict_proba(encoded_features)
    except AttributeError:
        probability = None  

    return prediction, probability

# User input
st.subheader("Enter Client Information")
country = st.selectbox("Select Country", options=["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.number_input("Year", min_value=2000, max_value=2023)
location_type = st.selectbox("Location Type", options=["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Access", options=["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, max_value=25)
relationship_with_head = st.selectbox("Relationship with Head", options=["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"])
education_level = st.selectbox("Education Level", options=["No formal education", "Primary", "Secondary", "Vocational/Specialised training ", "Tertiary", "Other/Dont know/RTA"])
job_type = st.selectbox("Job Type", options=["Farming and Fishing", "Self employed", "Formally employed Private", "Government Dependent", "Informally employed", "No Income", "Formally employed Government", "Remittance Dependent", "Other income", "Dont Know/Refuse to answer"])
age_group = st.selectbox("Age Group", options=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])

# Create a dataframe from the input
input_data = pd.DataFrame({
    'country': [country],
    'year': [year],
    'location_type': [location_type],
    'cellphone_access': [cellphone_access],
    'household_size': [household_size],
    'relationship_with_head': [relationship_with_head],
    'education_level': [education_level],
    'job_type': [job_type],
    'age_group': [age_group]
})

# Prediction button
if st.button("Predict"):
    result, probability = make_prediction(input_data)
    if result is not None:
        st.success(f'Bank account prediction: {"Yes" if result[0] == 1 else "No"}')
        if probability is not None:
            st.write(f'Probability: {probability[0]}')
    else:
        st.warning("No prediction available.")
