import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("Diabetes")

# Input fields for each parameter
Glucose = st.number_input("Glucose", min_value=0,max_value=200, value=0)
BloodPressure = st.number_input("BloodPressure", min_value=0,max_value=122, value=0)
Insulin = st.number_input("Insulin ", min_value=0, max_value=846, value=0 )
BMI= st.number_input("BMI", min_value=0, max_value=67, value=0.1)
DiabetesPedigreeFunction= st.number_input("DiabetesPedigreeFunction", min_value=0, max_value=2, value=0.1)
Age = st.number_input("Age", min_value=21, max_value=81, value=21)



# Create the input dictionary for prediction
input_data = {'Glucose': Glucose,
 'BloodPressure': BloodPressure,
 'Insulin': Insulin,
 'BMI':BMI,
 'DiabetesPedigreeFunction':DiabetesPedigreeFunction ,
 'Age':Age 
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_Outcome(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: Outcome with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: Did not Outcome with probability {prob:.2f}")
