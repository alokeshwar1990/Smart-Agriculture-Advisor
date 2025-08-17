import os
import zipfile
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ==============================
# File Paths (local in src/)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_DATA_ZIP = os.path.join(BASE_DIR, "crop_recommendation_dataset.csv.zip")
YIELD_DATA_ZIP = os.path.join(BASE_DIR, "crop_yield.csv.zip")

CROP_MODEL_FILE = os.path.join(BASE_DIR, "crop_model.pkl")
YIELD_MODEL_FILE = os.path.join(BASE_DIR, "yield_model.pkl")
CROP_ENCODERS_FILE = os.path.join(BASE_DIR, "crop_encoders.pkl")
YIELD_ENCODERS_FILE = os.path.join(BASE_DIR, "yield_encoders.pkl")

# ----------------------
# User data file
# ----------------------
USER_FILE = os.path.join(BASE_DIR, "users.pkl")
if os.path.exists(USER_FILE):
    with open(USER_FILE, "rb") as f:
        users = pickle.load(f)
else:
    users = {}  # format: {username: password}

# ==============================
# Prepare Models
# ==============================
def prepare_models():
    if all(os.path.exists(f) for f in [CROP_MODEL_FILE, YIELD_MODEL_FILE, CROP_ENCODERS_FILE, YIELD_ENCODERS_FILE]):
        return

    # Crop Recommendation Model
    with zipfile.ZipFile(CROP_DATA_ZIP, "r") as z:
        z.extractall(BASE_DIR)
    crop_df = pd.read_csv(os.path.join(BASE_DIR, "crop_recommendation_dataset.csv"))

    crop_features = ["Temperature", "Humidity", "Rainfall", "PH",
                     "Nitrogen", "Phosphorous", "Potassium", "Carbon", "Soil"]
    crop_target = "Crop"

    crop_encoders = {}
    if "Soil" in crop_df.columns:
        enc = LabelEncoder()
        crop_df["Soil"] = enc.fit_transform(crop_df["Soil"])
        crop_encoders["Soil"] = enc

    X_crop = crop_df[crop_features]
    y_crop = crop_df[crop_target]

    crop_encoders["Crop"] = LabelEncoder()
    y_crop = crop_encoders["Crop"].fit_transform(y_crop)

    X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
    crop_model = RandomForestClassifier(random_state=42)
    crop_model.fit(X_train, y_train)

    with open(CROP_MODEL_FILE, "wb") as f:
        pickle.dump(crop_model, f)
    with open(CROP_ENCODERS_FILE, "wb") as f:
        pickle.dump(crop_encoders, f)

    # Yield Prediction Model
    with zipfile.ZipFile(YIELD_DATA_ZIP, "r") as z:
        z.extractall(BASE_DIR)
    yield_df = pd.read_csv(os.path.join(BASE_DIR, "crop_yield.csv"))

    yield_features = ["Crop", "Crop_Year", "Season", "State", "Area",
                      "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
    yield_target = "Yield"

    yield_encoders = {}
    for col in ["Crop", "Season", "State"]:
        enc = LabelEncoder()
        yield_df[col] = enc.fit_transform(yield_df[col])
        yield_encoders[col] = enc

    X_yield = yield_df[yield_features]
    y_yield = yield_df[yield_target]

    X_train, X_test, y_train, y_test = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)
    yield_model = RandomForestRegressor(random_state=42)
    yield_model.fit(X_train, y_train)

    with open(YIELD_MODEL_FILE, "wb") as f:
        pickle.dump(yield_model, f)
    with open(YIELD_ENCODERS_FILE, "wb") as f:
        pickle.dump(yield_encoders, f)

# ==============================
# Load Models
# ==============================
def load_models():
    with open(CROP_MODEL_FILE, "rb") as f:
        crop_model = pickle.load(f)
    with open(YIELD_MODEL_FILE, "rb") as f:
        yield_model = pickle.load(f)
    with open(CROP_ENCODERS_FILE, "rb") as f:
        crop_encoders = pickle.load(f)
    with open(YIELD_ENCODERS_FILE, "rb") as f:
        yield_encoders = pickle.load(f)
    return crop_model, yield_model, crop_encoders, yield_encoders

# ==============================
# Main App UI
# ==============================
def app_ui():
    st.set_page_config(page_title="🌾 Smart Agriculture Advisor", layout="wide")
    st.title("🌾 Smart Agriculture Advisor")

    tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "📊 Yield Prediction"])

    # --------------------------
    # Tab 1: Crop Recommendation
    # --------------------------
    with tab1:
        st.header("🌱 Crop Recommendation System")
        st.write("Enter soil and environmental parameters to get recommended crop.")

        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)
            humidity = st.number_input("💧 Humidity (%)", value=60.0)
            rainfall = st.number_input("🌧️ Rainfall (mm)", value=200.0)
        with col2:
            ph = st.number_input("⚗️ Soil pH", value=6.5)
            nitrogen = st.number_input("🧪 Nitrogen (N)", value=50.0)
            phosphorous = st.number_input("🧪 Phosphorous (P)", value=30.0)
        with col3:
            potassium = st.number_input("🧪 Potassium (K)", value=40.0)
            carbon = st.number_input("🌍 Carbon (%)", value=1.2)

            # Dropdown for Soil types
            _, _, crop_encoders, _ = load_models()
            soil_types = crop_encoders["Soil"].classes_
            soil = st.selectbox("🪨 Soil Type", soil_types)

        if st.button("🔍 Recommend Crop"):
          crop_model, _, crop_encoders, _ = load_models()
          soil_val = crop_encoders["Soil"].transform([soil])[0]

          features = [[temperature, humidity, rainfall, ph,
                 nitrogen, phosphorous, potassium, carbon, soil_val]]

    # Make prediction
          prediction = crop_model.predict(features)

    # If prediction is already the crop name, just use it
          crop_name = prediction[0]

          st.success(f"🌱 Recommended Crop: **{crop_name}**")


    # --------------------------
    # Tab 2: Yield Prediction
    # --------------------------
    with tab2:
      st.header("📊 Crop Yield Prediction")
      st.write("Enter agricultural parameters to predict yield (tons/hectare).")

    # Load models & encoders first (to fetch dropdown options)
      _, yield_model, _, yield_encoders = load_models()

      col1, col2, col3 = st.columns(3)
      with col1:
        # Dropdown for Crop
        crop = st.selectbox(
            "🌱 Crop",
            options=list(yield_encoders["Crop"].classes_),
            index=0
        )

        crop_year = st.number_input("📅 Crop Year", value=2020, step=1)

        # Dropdown for Season
        season = st.selectbox(
            "☀️ Season",
            options=list(yield_encoders["Season"].classes_),
            index=0
        )

      with col2:
        # Dropdown for State
        state = st.selectbox(
            "🏞️ State",
            options=list(yield_encoders["State"].classes_),
            index=0
        )

        area = st.number_input("🌾 Area (hectares)", value=1.0)
        production = st.number_input("🏭 Production (tons)", value=2.5)

      with col3:
        annual_rainfall = st.number_input("🌧️ Annual Rainfall (mm)", value=800.0)
        fertilizer = st.number_input("🧴 Fertilizer (kg)", value=100.0)
        pesticide = st.number_input("🧪 Pesticide (kg)", value=10.0)

      if st.button("📈 Predict Yield"):
        # Encode categorical values
        crop_val = yield_encoders["Crop"].transform([crop])[0]
        season_val = yield_encoders["Season"].transform([season])[0]
        state_val = yield_encoders["State"].transform([state])[0]

        # Features
        features = [[
            crop_val, crop_year, season_val, state_val,
            area, production, annual_rainfall, fertilizer, pesticide
        ]]

        # Predict
        prediction = yield_model.predict(features)
        prediction_value = float(prediction.item())

        st.success(f"📊 Predicted Yield: **{prediction_value:.2f} tons/hectare**")


# ==============================
# Login / Sign Up UI
# ==============================
def login_ui():
    st.set_page_config(page_title="🌾 Smart Agriculture Advisor Login", layout="centered")
    st.title("🔐 Smart Agriculture Advisor Login")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""

    option = st.selectbox("Choose Action", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Sign Up":
        if st.button("Create Account"):
            if username in users:
                st.error("Username already exists!")
            elif username == "" or password == "":
                st.warning("Please enter username and password")
            else:
                users[username] = password
                with open(USER_FILE, "wb") as f:
                    pickle.dump(users, f)
                st.success("Account created! Please log in.")

    elif option == "Login":
        if st.button("Login"):
            if username in users and users[username] == password:
                st.success(f"Welcome, {username}!")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
            else:
                st.error("Invalid username or password")

    if st.session_state["logged_in"]:
        st.write(f"👋 Hello {st.session_state['username']}! Access your Smart Agriculture Advisor below.")
        app_ui()

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    prepare_models()
    login_ui()
