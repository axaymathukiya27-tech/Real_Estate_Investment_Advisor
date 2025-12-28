"""Real Estate Investment Advisor - Streamlit App (FIXED VERSION)."""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Add paths
sys.path.append(os.path.abspath("."))
from src.models import load_trained_models, make_predictions
from app.validation import InputValidator
from src.config import BEST_CLASSIFIER, BEST_REGRESSOR

# ==================== CONFIG ====================

# make DATA_DIR robust: resolve relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

CLF_MODEL_PATH = MODEL_DIR / BEST_CLASSIFIER
REG_MODEL_PATH = MODEL_DIR / BEST_REGRESSOR
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
PROCESSED_DATA_PATH = DATA_DIR / "housing_with_features.csv"

# Load feature config from canonical source (features package)
from src.features.build_features import load_feature_config
try:
    feature_config = load_feature_config()
except Exception as e:
    st.error(f"❌ Missing or invalid feature_config.json: {e}")
    st.stop()

NUMERIC_FEATURES = set(feature_config.get("numeric_features", []))
CATEGORICAL_FEATURES = set(feature_config.get("categorical_features", []))

# ==================== STREAMLIT CONFIG ====================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CACHED LOADERS ====================

@st.cache_data
def load_processed_data(path: str) -> pd.DataFrame:
    """Load processed dataset."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models."""

    try:
        if not CLF_MODEL_PATH.exists() or not REG_MODEL_PATH.exists():
            raise FileNotFoundError(f"Models not found in {MODEL_DIR}")

        clf_model, reg_model = load_trained_models(str(CLF_MODEL_PATH), str(REG_MODEL_PATH))
        return clf_model, reg_model

    except Exception as e:
        logging.exception(f"Error loading models: {e}")
        st.error(
            "❌ Critical: Could not load trained models.\n"
            "Make sure the model files exist under the `models/` directory and are readable.\n"
            "You can create them by running the hyperparameter tuning notebook or calling: `python -c \"from src.models import tune_models_with_random_search; tune_models_with_random_search(df)\"`\n"
            f"Error: {e}"
        )
        st.stop()
# ==================== MAIN APP ====================

def main():
    st.title("🏠 Real Estate Investment Advisor")
    st.write("🤖 AI-powered property investment predictions")
    
    # Load resources
    df = load_processed_data(str(PROCESSED_DATA_PATH))
    if df is None:
        st.stop()

    # Fail-fast check: ensure processed dataset contains all declared features
    declared_cols = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    missing = sorted([c for c in declared_cols if c not in df.columns])
    if missing:
        st.error(
            "❌ Processed dataset is missing features declared in feature_config.json: "
            f"{missing}.\nPlease re-run the feature pipeline or re-generate the processed dataset."
        )
        st.stop()

    clf_model, reg_model = load_models()
    
    # Get unique values for dropdowns
    states = sorted(df['State'].unique())
    
    # ==================== TABS ====================
    
    tab_predict, tab_insights = st.tabs(["🔮 Predict", "📊 Insights"])
    
    # ==================== TAB 1: PREDICTION ====================
    
    with tab_predict:
        with st.sidebar:
            st.title("📥 Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌍 Location")
                state = st.selectbox("State", states)
                cities = sorted(df[df['State'] == state]['City'].unique())
                city = st.selectbox("City", cities)
                localities = sorted(df[df['City'] == city]['Locality'].unique())
                locality = st.selectbox("Locality", localities[:50])
            
            with col2:
                st.markdown("### 🏢 Property")
                property_type = st.selectbox("Property Type", sorted(df['Property_Type'].unique()))
                bhk = st.slider("BHK", 1, 10, 3)
                total_floors = st.slider("Total Floors", 1, 50, 15)
                floor_no = st.slider("Floor No", 0, 50, 5)
            
            st.divider()
            
            st.markdown("### 📏 Size & Price")
            size_sqft = st.slider("Size (SqFt)", 300, 50000, 2000, step=50)
            price_lakhs = st.slider("Price (Lakhs)", 10, 500, 250, step=10)
            year_built = st.slider("Year Built", 1950, 2025, 2010)
            
            st.divider()
            
            st.markdown("### 🏫 Neighborhood")
            nearby_schools = st.slider("Nearby Schools", 0, 30, 5)
            nearby_hospitals = st.slider("Nearby Hospitals", 0, 30, 5)
            
            st.divider()
            
            st.markdown("### ⚙️ Other")
            furnished = st.selectbox("Furnished", sorted(df['Furnished_Status'].unique()))
            availability = st.selectbox("Availability", sorted(df['Availability_Status'].unique()))
            transport = st.selectbox("Public Transport", sorted(df['Public_Transport_Accessibility'].unique()))
            parking = st.selectbox("Parking", sorted(df['Parking_Space'].unique()))
            security = st.selectbox("Security", sorted(df['Security'].unique()))
            amenities = st.selectbox("Amenities", sorted(df['Amenities'].unique()))
            facing = st.selectbox("Facing", sorted(df['Facing'].unique()))
            owner_type = st.selectbox("Owner Type", sorted(df['Owner_Type'].unique()))
            
            predict_btn = st.button("🔮 Predict", use_container_width=True)
        
        # ==================== VALIDATION & PREDICTION ====================
        
        if predict_btn:
            try:
                # Validate inputs
                is_valid, errors = InputValidator.validate(
                    bhk, size_sqft, price_lakhs, year_built,
                    floor_no, total_floors, nearby_schools, nearby_hospitals
                )
            
                if not is_valid:
                    st.error("❌ Input Validation Errors:")
                    for error in errors:
                        st.write(f"• {error}")
                    st.stop()

                # ---------- Derived / Engineered Features ----------
                current_year = 2025
                age_of_property = current_year - year_built
                price_per_sqft = price_lakhs / size_sqft if size_sqft > 0 else 0

                # Encodings (must match training logic)
                furnished_map = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
                availability_map = {"Under_Construction": 0, "Ready_to_Move": 1}
                level_map = {"Low": 0, "Medium": 1, "High": 2}

                furnished_enc = furnished_map.get(furnished, 1)
                availability_enc = availability_map.get(availability, 1)
                transport_score = level_map.get(transport, 1)
                security_score = level_map.get(security, 1)

                # City-level statistics (fallback-safe): only use these if they exist in
                # the processed dataset and are declared in the feature config.
                city_data = df[df["City"] == city]

                if "Annual_Growth_Rate" in df.columns and "Annual_Growth_Rate" in NUMERIC_FEATURES:
                    annual_growth = city_data["Annual_Growth_Rate"].median() if len(city_data) > 0 else 0.05
                else:
                    annual_growth = None

                if "Investment_Score" in df.columns and "Investment_Score" in NUMERIC_FEATURES:
                    investment_score = city_data["Investment_Score"].median() if len(city_data) > 0 else 3
                else:
                    investment_score = None

                # Build input ONLY with features declared in feature_config.json to guarantee
                # identical feature sets between training, tuning and inference.
                input_dict = {}

                # Numeric features
                # Map UI values to the expected numeric features if present in config
                numeric_map = {
                    "BHK": bhk,
                    "Size_in_SqFt": size_sqft,
                    "Price_in_Lakhs": price_lakhs,
                    "Year_Built": year_built,
                    "Floor_No": floor_no,
                    "Total_Floors": total_floors,
                    "Nearby_Schools": nearby_schools,
                    "Nearby_Hospitals": nearby_hospitals,
                    "Age_of_Property": age_of_property,
                    "Price_per_SqFt": price_per_sqft,
                    "Annual_Growth_Rate": annual_growth,
                    "Investment_Score": investment_score,
                    "Furnished_Status_Enc": furnished_enc,
                    "Availability_Status_Enc": availability_enc,
                    "Transport_Score": transport_score,
                    "Security_Score": security_score,
                }

                for k, v in numeric_map.items():
                    if k in NUMERIC_FEATURES and v is not None:
                        input_dict[k] = v

                # Categorical features
                categorical_map = {
                    "State": state,
                    "City": city,
                    "Locality": locality,
                    "Property_Type": property_type,
                    "Furnished_Status": furnished,
                    "Availability_Status": availability,
                    "Public_Transport_Accessibility": transport,
                    "Parking_Space": parking,
                    "Security": security,
                    "Amenities": amenities,
                    "Facing": facing,
                    "Owner_Type": owner_type,
                }

                for k, v in categorical_map.items():
                    if k in CATEGORICAL_FEATURES:
                        input_dict[k] = v

                
                input_df = pd.DataFrame([input_dict])
                
                # Predict
                with st.spinner("🔄 Running predictions..."):
                    pred_label, pred_proba, pred_price = make_predictions(clf_model, reg_model, input_df)
                
                is_good = bool(pred_label.iloc[0])
                prob = float(pred_proba.iloc[0])
                future_price = float(pred_price.iloc[0 ])

                # ---------- Risk Level (Cosmetic) ----------
                if prob >= 0.85:
                    risk = "🟢 Low Risk"
                elif prob >= 0.65:
                    risk = "🟡 Medium Risk"
                else:
                    risk = "🔴 High Risk"
   
                # Display results
                st.markdown("""
                <div style="
                    padding:20px;
                    border-radius:12px;
                    background: linear-gradient(135deg, #1f2937, #111827);
                    border:1px solid #374151;
                ">
                <h2>🏠 Investment Verdict</h2>
                <p style="font-size:18px;">AI-based recommendation for this property</p>
                </div>
                """, unsafe_allow_html=True)
                st.subheader("🔮 Prediction Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Good Investment?",
                        "✅ YES" if is_good else "❌ NO"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{prob*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Risk Level",
                        risk
                    )
                
                with col4:
                    st.metric(
                        "5-Year Price",
                        f"₹{future_price:.2f}L"
                    )
                
                # Details
                roi_percent = ((future_price - price_lakhs) / price_lakhs) * 100
                
                with st.expander("📊 Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estimated ROI", f"{roi_percent:.1f}%")
                    with col2:
                        st.metric("Price/SqFt", f"₹{future_price / (size_sqft/10000):.0f}")
                
                st.success("✓ Prediction complete!")
            
            except Exception as e:
                logging.exception("Prediction failed during user request")
                import traceback
                st.error(f"❌ Prediction error: {e}")
                st.error(traceback.format_exc())
    
    # ==================== TAB 2: INSIGHTS ====================
    
    with tab_insights:
        st.subheader("📈 Market Insights")
        
        all_cities = ["All"] + sorted(df['City'].unique().tolist())
        selected_city = st.selectbox("Filter by City", all_cities, index=0)
        
        if selected_city != "All":
            df_insights = df[df['City'] == selected_city]
        else:
            df_insights = df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏙️ Avg Price/SqFt by City")
            city_price = df_insights.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False).head(10)
            st.bar_chart(city_price)
        
        with col2:
            st.markdown("#### ✅ Good Investment Rate")
            if "Good_Investment" in df_insights.columns:
                city_good = df_insights.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(10)
                st.bar_chart(city_good)
        
st.markdown("---")
st.caption("Built by Axay | Real Estate Investment Advisor • ML + Streamlit")

if __name__ == "__main__":
    main()
