# app.py
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🌱 Smart Agriculture Advisor", layout="wide")

# ---- Artifact names ----
CROP_MODEL = "crop_model.pkl"
CROP_FEATURES = "crop_features.pkl"
CROP_FEAT_ENCODERS = "crop_feature_encoders.pkl"
CROP_LABEL_ENCODER = "crop_label_encoder.pkl"

YIELD_MODEL = "yield_model.pkl"
YIELD_FEATURES = "yield_features.pkl"
YIELD_FEAT_ENCODERS = "yield_feature_encoders.pkl"

# ---- Auto-train if missing (optional) ----
def auto_train_if_missing():
    missing = [p for p in [CROP_MODEL, CROP_FEATURES, CROP_FEAT_ENCODERS, YIELD_MODEL, YIELD_FEATURES, YIELD_FEAT_ENCODERS] if not os.path.exists(p)]
    if missing:
        st.warning("📦 Training artifacts missing. Running trainer once...")
        os.system("python train_models.py")  # assumes same folder

auto_train_if_missing()

# ---- Load artifacts ----
try:
    crop_model = joblib.load(CROP_MODEL)
    crop_features = joblib.load(CROP_FEATURES)
    crop_feat_encoders = joblib.load(CROP_FEAT_ENCODERS)
    crop_label_encoder = joblib.load(CROP_LABEL_ENCODER) if os.path.exists(CROP_LABEL_ENCODER) else None

    yield_model = joblib.load(YIELD_MODEL)
    yield_features = joblib.load(YIELD_FEATURES)
    yield_feat_encoders = joblib.load(YIELD_FEAT_ENCODERS)
except Exception as e:
    st.error(f"❌ Failed to load models/encoders: {e}")
    st.stop()

st.title("🌾 Smart Agriculture Advisor")

tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "📊 Yield Prediction"])

# ---------- Helpers ----------
def build_input_row(feature_names, encoders_dict):
    """
    Returns (values_dict, ui_widgets_dict).
    For categorical features (having an encoder), shows a selectbox.
    For numeric features, shows a number_input.
    """
    values = {}
    for feat in feature_names:
        if feat in encoders_dict:
            classes = encoders_dict[feat].classes_
            sel = st.selectbox(f"{feat}", classes)
            values[feat] = encoders_dict[feat].transform([sel])[0]
        else:
            # numeric
            values[feat] = st.number_input(f"{feat}", value=0.0)
    return values

def df_from_values(values: dict, ordered_cols: list) -> pd.DataFrame:
    df = pd.DataFrame([values])
    # Ensure exact column order & presence
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = 0
    return df[ordered_cols]

# ---------- TAB 1: Crop Recommendation ----------
with tab1:
    st.subheader("Enter field parameters")
    with st.form("crop_form"):
        crop_values = build_input_row(crop_features, crop_feat_encoders)
        submit_crop = st.form_submit_button("🔍 Recommend Crop")

    if submit_crop:
        X = df_from_values(crop_values, crop_features)
        try:
            pred = crop_model.predict(X)[0]
            if crop_label_encoder is not None:
                pred = crop_label_encoder.inverse_transform([pred])[0]
            st.success(f"✅ Recommended Crop: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- TAB 2: Yield Prediction ----------
with tab2:
    st.subheader("Enter parameters for yield prediction")
    with st.form("yield_form"):
        yield_values = build_input_row(yield_features, yield_feat_encoders)
        submit_yield = st.form_submit_button("📈 Predict Yield")

    if submit_yield:
        Xy = df_from_values(yield_values, yield_features)
        try:
            yhat = yield_model.predict(Xy)[0]
            st.info(f"🌱 Predicted Yield: **{yhat:.2f}** (units of your dataset)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
