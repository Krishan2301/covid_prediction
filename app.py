import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("corona.csv")

# Train the model (you can also save and load a pre-trained model using joblib)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LogisticRegression()
model.fit(X, y)

# App title
st.title("COVID Infection Prediction App")

# User input form
st.header("Enter the details to predict infection:")

# Dynamically create input fields based on feature columns
input_data = {}
for col in X.columns:
    dtype = X[col].dtype
    if np.issubdtype(dtype, np.number):
        value = st.number_input(f"{col}", value=float(X[col].mean()))
    else:
        value = st.text_input(f"{col}", value=str(X[col].iloc[0]))
    input_data[col] = value

# Prediction
if st.button("Predict"):
    user_df = pd.DataFrame([input_data])
    prediction = model.predict(user_df)[0]
    st.subheader("Prediction Result:")
    st.write("✅ Infected" if prediction == 1 else "❌ Not Infected")

