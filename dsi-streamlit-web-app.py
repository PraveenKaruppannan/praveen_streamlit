# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:47:24 2023

@author: PKA232
"""

#import Libraries

import streamlit as st
import pandas as pd 
import joblib 

#loading our model pipeline object 

model = joblib.load("model.joblib")

#add title and instructions

st.title("Purchase predection Model")
st.subheader("Enter customer information and sumbit for likelihood to purchase")

# age input form

age = st.number_input(
    label= "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)

# gender import form 

gender = st.radio(
    label= "02. Enter the customer's gender",
    options = ["M","F"])

# Credit Score input form 

credit_score = st.number_input(
    label= "01. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500)


# submit input to model 

if st.button("Submit For Predection"):
    
    #store our data in dataframe 
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # apply model pipeline to the input daat and extreact probability predection
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output predection
    st.subheader(f"Based in the customer attribute, our model predicts a purchase probability of {pred_proba:.0%}")
























