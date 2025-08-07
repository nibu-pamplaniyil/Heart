import streamlit as st
import pickle
import numpy as np

# Load the model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page title
st.title("Heart Disease Prediction App")

# User input for model features
st.header("Enter Patient Details:")

age = st.slider("Age", 20, 90, 50)
sex = st.radio("Sex", ["Male", "Female"])
ind_typ_angina = st.selectbox("Typical Angina", [0, 1])
ind_atyp_angina = st.selectbox("Atypical Angina", [0, 1])
ind_non_ang_pain = st.selectbox("Non-Anginal Pain", [0, 1])
resting_BP = st.slider("Resting Blood Pressure", 80, 200, 120)
Serum_cholest = st.slider("Serum Cholesterol", 100, 600, 200)
blood_sugar_exc120 = st.selectbox("Blood Sugar > 120 mg/dL", [0, 1])
ind_for_ecg_1 = st.selectbox("ECG: ST-T Abnormality", [0, 1])
ind_for_ecg_2 = st.selectbox("ECG: Left Ventricular Hypertrophy", [0, 1])
Max_heart_rate = st.slider("Max Heart Rate Achieved", 60, 220, 150)
ind_exerc_angina = st.selectbox("Exercise-Induced Angina", [0, 1])
ST_dep_by_exerc = st.slider("ST Depression (Exercise)", 0.0, 6.0, 1.0)
ind_for_slope_up_exerc = st.selectbox("Slope: Upsloping", [0, 1])
ind_for_slope_down_exerc = st.selectbox("Slope: Downsloping", [0, 1])
num_vessels_fluro = st.selectbox("No. of Vessels (Flourosopy)", [0, 1, 2, 3])
Thal_rev_defect = st.selectbox("Thal: Reversible Defect", [0, 1])
Thal_fixed_defect = st.selectbox("Thal: Fixed Defect", [0, 1])

# Convert sex to binary
sex = 1 if sex == "Male" else 0

# Prepare input array
input_data = np.array([[age, sex, ind_typ_angina, ind_atyp_angina, ind_non_ang_pain,
                        resting_BP, Serum_cholest, blood_sugar_exc120, ind_for_ecg_1,
                        ind_for_ecg_2, Max_heart_rate, ind_exerc_angina, ST_dep_by_exerc,
                        ind_for_slope_up_exerc, ind_for_slope_down_exerc, num_vessels_fluro,
                        Thal_rev_defect, Thal_fixed_defect]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result:")
    st.write("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
    st.write(f"Probability of Heart Disease: {probability:.2f}")
