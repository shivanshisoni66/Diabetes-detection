import joblib 
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
with open('diabetes_model.pkl', 'rb') as file: 
  model = joblib.load(file) 
#• Opens the saved XGBoost model file (XGBoost.pkl). 
#• 'rb' → means read in binary mode. 
#• joblib.load(file) → loads the trained model so we can use it for predictions. 
st.set_page_config( 
page_title="Diabetes Detection Dashboardd", 
page_icon="🩺", 
layout="wide", 
initial_sidebar_state="expanded" 
) 
#Sets the title of the web page, the emoji icon, and the layout style. 
layout="wide" #→ makes the app use full screen width. 
  #expanded → sidebar is open by default. 
 
 
st.markdown(""" 
<style> 
body { 
    background-image: url(''); 
    background-size: cover; 
    background-attachment: fixed; 
    background-position: center; 
} 
.main { 
    background-color: rgba(255, 255, 255, 0.88); 
    padding: 2rem; 
    border-radius: 12px; 
    margin: 2rem; 
} 
.prediction-card { 
    border-radius: 12px; 
    padding: 25px; 
    margin: 20px 0; 
    box-shadow: 0 6px 12px rgba(0,0,0,0.15); 
    background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%); 
    color: white; 
    border-left: 6px solid #ffab00; 
    transition: transform 0.3s ease, box-shadow 0.3s ease; 
} 
.prediction-card:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 8px 16px rgba(0,0,0,0.2); 
} 
.prediction-card h3 { 
    margin-top: 0; 
    font-size: 1.5rem; 
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2); 
} 
.prediction-card p { 
    margin-bottom: 0.5rem; 
    font-size: 1.1rem; 
    opacity: 0.9; 
} 
.high-risk { 
    color: #ffeb3b; 
    font-weight: 800; 
    text-shadow: 0 0 8px rgba(255,235,59,0.4); 
} 
.low-risk { 
    color: #69f0ae; 
    font-weight: 800; 
    text-shadow: 0 0 8px rgba(105,240,174,0.3); 
} 
.recommendation { 
    background-color: rgba(255,255,255,0.15); 
    padding: 12px; 
    border-radius: 8px; 
    margin-top: 15px; 
    font-weight: 500; 
} 
.feature-importance { 
    background: white; 
    border-radius: 10px; 
    padding: 15px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
} 
.stProgress > div > div > div { 
    background-color: #2e86ab; 
} 
</style> 
""", unsafe_allow_html=True) 
 
 
 
 
 
st.title("🩺 DiaTrack") 
st.subheader("*A machine learning application predicting diabetes risk based "
"on clinical parameters*") 
 
 
 
 
with st.form("patient_form"): 
 st.subheader("🧾 Patient Clinical Parameters") 
#Creates a form where the user can input their health values. 
#The form waits for the submit button before sending the values to the model. 
col1, col2 = st.columns(2) 
# Splits the form into two columns for better layout. 
pregnancies = st.slider("Number of pregnancies", 0, 17, 2) 
glucose = st.slider("Glucose level (mg/dL)", 0, 200, 117) 
blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 200, 72) 
skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 23) 
#User selects Pregnancies, Glucose, BP, Skin Thickness using sliders. 
insulin = st.slider("Insulin (μU/mL)", 0, 846, 30) 
bmi = st.slider("BMI (kg/m²)", 0.0, 67.1, 32.0, 0.1) 
diabetes_pedigree = st.slider("Diabetes Pedigree", 0.078, 2.42, 0.372, 0.001) 
age = st.slider("Age (years)", 21, 81, 29) 
#User selects Insulin, BMI, Diabetes Pedigree Function, Age. 
submitted = st.form_submit_button("Predict Diabetes Risk", type="primary") 
 
if submitted and model: 
    input_data = np.array([[pregnancies, glucose, blood_pressure, 
skin_thickness, 
                            insulin, bmi, diabetes_pedigree, age]]) 
 
  #Checks if the button was clicked. 
  #Collects all inputs into a NumPy array (so the ML model can read them). 
 
 
 
prediction = model.predict(input_data)[0] 
proba = model.predict_proba(input_data)[0] 
 
 # model.predict() #→ tells whether the patient is Diabetic (1) or Not Diabetic (0). 
  #model.predict_proba()# → gives the probability (%) for each class. 
 
 
 
 
if prediction == 1: 
    st.markdown(f"""
        <div class="prediction-card"> 
            <h3 class="high-risk">⚠️ HIGH RISK OF DIABETES</h3> 
            <p>Probability: <strong>{proba[1]*100:.1f}%</strong></p> 
            <p class="recommendation">Recommendation: Please consult with a 
healthcare provider.</p> 
        </div> 
    """) 
else: 
    st.markdown(f""" 
        <div class="prediction-card"> 
            <h3 class="low-risk">✅ LOW RISK OF DIABETES</h3> 
            <p>Probability: <strong>{proba[0]*100:.1f}%</strong></p> 
            <p class="recommendation">Recommendation: Maintain a healthy 
lifestyle.</p> 
        </div> 
    """, unsafe_allow_html=True) 
  #If prediction = 1 → shows High Risk card. 
  #If prediction = 0 #→ shows Low Risk card. 
  #Probability is displayed (e.g., 75.3%). 
 
 
 
 
features = ['Pregnancies', 'Glucose', 'BP', 'Skin Thickness', 'Insulin', 'BMI', 
'Pedigree', 'Age'] 
importances = model.feature_importances_ 
#Gets importance scores of each feature (how much it contributed to prediction). 
 
 
 
ax.barh(y_pos, importances, align='center', color='#2e86ab') 
st.pyplot(fig)
#Plots a horizontal bar chart showing the most important health parameters. 
 
with st.sidebar: 
 st.image("image_link_here", width=180) 
st.markdown("---") 
st.markdown("Built with ❤️ for the competition") 
st.subheader("📈 Model Performance Metrics") 
col1, col2, col3 = st.columns(3) 
with col1: 
 st.metric("Accuracy", "79.87%") 
with col2: 
 st.metric("Precision", "74.47%", help="Correct positive predictions") 
with col3: 
 st.metric("Recall", "64.81%", help="True positives identified") 
#Shows evaluation metrics of the trained model: 
#• Accuracy → Overall correct predictions. 
#• Precision → Out of predicted diabetics, how many were correct. 
#• Recall → Out of actual diabetics, how many were correctly identified. 
 
 
