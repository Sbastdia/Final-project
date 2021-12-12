import pickle
import numpy as np
import pandas as pd
import streamlit as st

# loading the model
models_path = 'C:/Users/Usuario/finalProject/savemodels/'
model_name = models_path + 'linearmodel.pkl'
loaded_model = pickle.load(open(model_name, 'rb'))

# loading the scaler
scalers_path = 'scalers/'
scaler_name = scalers_path + 'scaler.pkl'
loaded_scaler = pickle.load(open(scaler_name, 'rb'))

#############
# Main page #
#############
st.write("The model for diabetes prediction")


# Get input values - numeric variables

times_pregnant = st.number_input("Please enter the number of times beeing pregnant: ")
glucose_concentration = st.number_input("Please enter the lasma glucose concentration a 2 hours in an oral glucose tolerance test: ")
blood_pressure = st.number_input("Please enter the diastolic blood pressure (mm Hg): ")
skin_thickness = st.number_input("Please enter the triceps skin fold thickness (mm): ")
serum_insulin = st.number_input("Please enter the 2-Hour serum insulin (mu U/ml): ")
bmi = st.number_input("Please enter the body mass index (weight in kg/(height in m)^2): ")
pedigree = st.number_input("Please enter the diabetes pedigree function: ")
age = st.number_input("Please enter the age: ")




# when 'Predict' is clicked, make the prediction and store it
if st.button("Get Your Prediction"):

    X = pd.DataFrame({'times_pregnant':[times_pregnant],
                    'glucose_concentration':[glucose_concentration],
                    'blood_pressure':[blood_pressure],
                    'skin_thickness':[skin_thickness],
                    'serum_insulin':[serum_insulin],
                    'bmi':[bmi],
                    'pedigree':[pedigree],
                    'age':[age]
                    })

    # Scaling data
    X_scaled = loaded_scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

    # Making predictions
    prediction = loaded_model.predict(X_scaled_df)
    #prediction = loaded_model.predict_proba(X)[:,1] # The model produces (p0,p1), we want p1.

    if ( prediction == 0 ):
        st.success("The prediction of the model is: you are healthy!!!")
    else:
        st.success("The prediction of the model is: you have diabetes")