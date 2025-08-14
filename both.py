import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
CROP_DATA_PATH = BASE_DIR / "crop_recommendation_dataset.csv"
YIELD_DATA_PATH = BASE_DIR / "crop_yield.csv"

CROP_MODEL_PATH = BASE_DIR / "crop_model.pkl"
YIELD_MODEL_PATH = BASE_DIR / "yield_model.pkl"
ENCODERS_PATH = BASE_DIR / "feature_encoders.pkl"

# ---------------- TRAINING FUNCTIONS ----------------
@st.cache_resource
def train_crop_model():
    df = pd.read_csv(CROP_DATA_PATH)
    
    label_encoders = {}
    
    # Encode categorical columns
    for col in ['Soil', 'Crop']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df[['Temperature', 'Humidity', 'Rainfall', 'PH', 
            'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon', 'Soil']]
    y = df['Crop']
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, CROP_MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    
    return model, label_encoders

@st.cache_resource
def train_yield_model():
    df = pd.read_csv(YIELD_DATA_PATH)
    
    label_encoders = {}
    for col in ['Crop', 'Season', 'State']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 
            'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
    y = df['Yield']
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, YIELD_MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    
    return model, label_encoders

# ---------------- LOAD OR TRAIN ----------------
def load_or_train():
    if CROP_MODEL_PATH.exists() and YIELD_MODEL_PATH.exists() and ENCODERS_PATH.exists():
        crop_model = joblib.load(CROP_MODEL_PATH)
        yield_model = joblib.load(YIELD_MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
    else:
        st.info("Training models for the first time... please wait ⏳")
        crop_model, encoders = train_crop_model()
        yield_model, encoders_y = train_yield_model()
        encoders.update(encoders_y)
    return crop_model, yield_model, encoders

crop_model, yield_model, encoders = load_or_train()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Smart Agriculture Advisor", layout="wide")

st.title("🌾 Smart Agriculture Advisor")
tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "📈 Yield Prediction"])

# ---------------- TAB 1: Crop Recommendation ----------------
with tab1:
    st.subheader("Enter Environmental & Soil Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
        phosphorous = st.number_input("Phosphorous (P)", 0.0, 200.0, 50.0)
    with col2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
        potassium = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
        carbon = st.number_input("Carbon (%)", 0.0, 10.0, 1.0)
    with col3:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
        ph = st.number_input("pH", 0.0, 14.0, 6.5)
        soil = st.selectbox("Soil Type", encoders['Soil'].classes_)

    if st.button("Recommend Crop"):
        soil_encoded = encoders['Soil'].transform([soil])[0]
        features = np.array([[temperature, humidity, rainfall, ph, 
                               nitrogen, phosphorous, potassium, carbon, soil_encoded]])
        prediction = crop_model.predict(features)[0]
        crop_name = encoders['Crop'].inverse_transform([prediction])[0]
        st.success(f"🌱 Recommended Crop: **{crop_name}**")

# ---------------- TAB 2: Yield Prediction ----------------
with tab2:
    st.subheader("Predict Crop Yield")

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox("Crop", encoders['Crop'].classes_)
        crop_year = st.number_input("Crop Year", 2000, 2050, 2024)
        season = st.selectbox("Season", encoders['Season'].classes_)
    with col2:
        state = st.selectbox("State", encoders['State'].classes_)
        area = st.number_input("Area (ha)", 0.0, 100000.0, 100.0)
        rainfall = st.number_input("Annual Rainfall (mm)", 0.0, 5000.0, 1000.0)
    with col3:
        fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 1000.0, 100.0)
        pesticide = st.number_input("Pesticide (kg/ha)", 0.0, 500.0, 10.0)

    if st.button("Predict Yield"):
        crop_encoded = encoders['Crop'].transform([crop])[0]
        season_encoded = encoders['Season'].transform([season])[0]
        state_encoded = encoders['State'].transform([state])[0]
        
        features = np.array([[crop_encoded, crop_year, season_encoded, state_encoded, 
                               area, rainfall, fertilizer, pesticide]])
        prediction = yield_model.predict(features)[0]
        st.success(f"📈 Predicted Yield: **{prediction:.2f} quintals/ha**")
