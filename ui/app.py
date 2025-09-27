import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Load the Model and Scaler ---
# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_model_and_scaler():
    """Load the saved model and scaler from disk."""
    try:
        model = joblib.load('final_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please run the `export_model.py` script first.")
        return None, None

model, scaler = load_model_and_scaler()

# --- Main App Interface ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title('🩺 Heart Disease Prediction App')
st.write("This app predicts the likelihood of a patient having heart disease based on their medical data. Please fill in the details below.")


if model is not None and scaler is not None:
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Patient Demographics")
        age = st.number_input('Age', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex', ['Male', 'Female'])

    with col2:
        st.header("Symptoms & Vitals")
        cp = st.selectbox('Chest Pain Type (CP)', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
        trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
        chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ['False', 'True'])

    with col3:
        st.header("Medical Test Results")
        restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', ['normal', 'st-t abnormality', 'lv hypertrophy'])
        thalch = st.number_input('Maximum Heart Rate Achieved (thalch)', min_value=60, max_value=220, value=150)
        exang = st.selectbox('Exercise Induced Angina (exang)', ['False', 'True'])
        oldpeak = st.number_input('ST depression induced by exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox('Slope of the peak exercise ST segment', ['upsloping', 'flat', 'downsloping'])
        ca = st.number_input('Number of major vessels colored by flourosopy (ca)', min_value=0, max_value=4, value=0)
        thal = st.selectbox('Thalassemia (thal)', ['normal', 'fixed defect', 'reversable defect'])

    # --- Prediction Logic ---
    if st.button('**Predict Heart Disease Risk**', type="primary"):
        # This is the full list of columns the model was trained on
        # It's crucial that the input data has the exact same columns in the same order
        model_columns = ['age', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'ca',
                         'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
                         'restecg_normal', 'restecg_st-t abnormality', 'slope_flat',
                         'slope_upsloping', 'thal_normal', 'thal_reversable defect']

        # Create a dictionary to hold user input, initialized to zero for all columns
        input_data = {col: 0 for col in model_columns}

        # Populate the dictionary with user inputs (numerical and direct boolean-like)
        input_data['age'] = age
        input_data['trestbps'] = trestbps
        input_data['chol'] = chol
        input_data['thalch'] = thalch
        input_data['oldpeak'] = oldpeak
        input_data['ca'] = ca
        input_data['fbs'] = 1 if fbs == 'True' else 0
        input_data['exang'] = 1 if exang == 'True' else 0

        # Handle one-hot encoded columns based on user selection
        if sex == 'Male':
            input_data['sex_Male'] = 1

        # Set the appropriate one-hot encoded column to 1 if it exists in our model columns
        cp_col = f'cp_{cp}'
        if cp_col in input_data:
            input_data[cp_col] = 1

        restecg_col = f'restecg_{restecg}'
        if restecg_col in input_data:
            input_data[restecg_col] = 1

        slope_col = f'slope_{slope}'
        if slope_col in input_data:
            input_data[slope_col] = 1

        thal_col = f'thal_{thal}'
        if thal_col in input_data:
            input_data[thal_col] = 1

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order is the same as the training data
        input_df = input_df[model_columns]

        # Scale the data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # --- Display Results ---
        st.write("---")
        st.header("Prediction Result")

        if prediction[0] == 1:
            st.error(f'**High Risk of Heart Disease** (Probability: {prediction_proba[0][1]*100:.2f}%)')
            st.write("Based on the provided data, the model indicates a high likelihood of heart disease. It is strongly recommended to consult a medical professional for a comprehensive evaluation.")
        else:
            st.success(f'**Low Risk of Heart Disease** (Probability: {prediction_proba[0][0]*100:.2f}%)')
            st.write("The model suggests a low likelihood of heart disease based on the input data. Maintaining a healthy lifestyle is always recommended.")

