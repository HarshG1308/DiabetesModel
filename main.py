import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data = pd.read_csv(r'diabetes_prediction_dataset.csv')

data['gender']=data['gender'].map({'Male':1, 'Female':2, 'Other':3})

data['smoking_history']=data['smoking_history'].map({'never':1, 'current':2, 'No Info':3,'former':4,'ever':5,'not current':6})

y=data['diabetes']
data.drop('diabetes',axis=1,inplace=True)
x=data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence
model.fit(X_train, Y_train)

import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Title of the app
st.title("Diabetes Prediction App")

# Create input fields for the user
gender = st.selectbox("Gender", ("Male", "Female", "Other"))
age = st.number_input("Age", min_value=0, max_value=120, value=25)
hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", (0, 1))
heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", (0, 1))
smoking_history = st.selectbox("Smoking History", ("never", "current", "formerly", "No Info"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100)

# Map gender to numeric values
gender_map = {'Male': 1, 'Female': 2, 'Other': 3}
gender_numeric = gender_map[gender]

# Map smoking history to numeric values
smoking_map = {'never': 0, 'current': 1, 'formerly': 2, 'No Info': 3}
smoking_numeric = smoking_map[smoking_history]

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'gender': [gender_numeric],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_numeric],
    'bmi': [bmi],
    'HbA1c_level': [hba1c_level],
    'blood_glucose_level': [blood_glucose_level]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("The model predicts: No Diabetes")
    else:
        st.error("The model predicts: Diabetes")