import streamlit as st
import joblib

st.title("Iris Flower Classification")

reloaded_joblib = joblib.load('iris_joblib')

# sliders for input features

sepal_length = st.slider("Sepal Length",4.3,7.9)
sepal_width = st.slider("Sepal Length",2.0,4.4)
petal_length = st.slider("Sepal Length",1.0,6.9)
petal_width = st.slider("Sepal Length",0.1,2.5)

y_pred = reloaded_joblib.predict([[sepal_length,sepal_width,petal_length,petal_width]])

if st.button('Predict'):
    #op = ['Sentosa','Versicolor','Virginica']
    #st.title(op(int(y_pred[0])))
    st.title(y_pred[0])
