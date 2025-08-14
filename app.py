import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
model = tf.keras.models.load_model("model.h5")
# load  the model and scalar and pickle files
with open("label_encoder_gender.pkl","rb") as f :
    label_encoder_gen = pickle.load(f)
with open("one_hot_encoder_geo.pkl","rb") as f :
    one_hot_encoder_geo = pickle.load(f)
with open("scaler.pkl","rb") as f :
    scaler = pickle.load(f)
st.title("customer churn prediction")

# User input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
geo_encoder = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoder,columns=one_hot_encoder_geo.get_feature_names_out(["Geography"]))
input_data1 = pd.concat([input_data.reset_index(drop = True),geo_df],axis=1)
scaled = scaler.transform(input_data1)
pred = model.predict(scaled)
if pred[0][0] > 0.5 :
    st.write("the user will churn")
else :
    st.write("user will not churn")