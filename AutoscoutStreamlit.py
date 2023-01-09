import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


#title text
st.title("Car Price Prediction")


make_model_selected=st.selectbox("Make-Model", ['Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia',
       'Renault Clio', 'Renault Duster', 'Renault Espace'])
km_selected = st.slider("KM", min_value=0, max_value=317000, value=100000, step=1000)
hp_selected = st.slider("horse Power (KW)", min_value=40, max_value=294, value=40, step=1)
age_selected = st.slider("Age", min_value=0, max_value=3, value=0, step=1)
gear_type_selected = st.selectbox("Gear Type", ['Automatic', 'Manual', 'Semi-automatic'])
gears_selected = st.slider("Gears", min_value=5, max_value=8, value=5, step=1)
type_selected=st.selectbox("Type", ['Used', "Employee's car", 'New', 'Demonstration', 'Pre-registered'])
Safety_Security_Package_selected=st.selectbox("Safety Package", ['Safety Premium Package', 'Safety Premium Plus Package',
       'Safety Standard Package'])

my_dict = {
    "make_model": make_model_selected,
    "km": km_selected,
    "hp_kW": hp_selected,
    "age": age_selected,
    "Gearing_Type": gear_type_selected,
    "Gears": gears_selected,
    "Type":type_selected,
    'Safety_Security_Package':Safety_Security_Package_selected
}


import pickle
filename = 'final_logistic_pipe_model'
model = pickle.load(open(filename, 'rb'))

df=pd.DataFrame.from_dict([my_dict])

st.table(df)

if st.button("Predict"):
    pred = model.predict(df)
    st.success("Predicted price is: {:.2f}".format(pred[0]))





