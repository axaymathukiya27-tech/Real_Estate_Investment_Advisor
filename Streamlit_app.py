import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ---- Page config (only once) ----
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
)

# Ensure src package is importable
sys.path.append(os.path.abspath("."))

from src.models import load_trained_models, make_predictions


@st.cache_data
def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_models() -> tuple:
    clf_path = os.path.join("models", "best_classification_model.pkl")
    reg_path = os.path.join("models", "best_regression_model.pkl")
    clf_model, reg_model = load_trained_models(clf_path, reg_path)
    return clf_model, reg_model


def main():
    st.title("🏠 Real Estate Investment Advisor")
    st.write(
        "Predict **future price** and **investment quality** for a housing property "
        "using machine learning models trained on an India housing dataset."
    )

    # ---------- Load data & models ----------
    data_path = os.path.join("data", "processed", "housing_with_features.csv")
    df = load_processed_data(data_path)
    clf_model, reg_model = load_models()

    # Features used by the models
    feature_numeric = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Furnished_Status_Enc",
        "Availability_Status_Enc",
        "Transport_Score",
        "Security_Score",
        "Annual_Growth_Rate",
        "Investment_Score",
    ]

    feature_categorical = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "Furnished_Status",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
    ]

    feature_cols = feature_numeric + feature_categorical

    # City-level stats for defaults
    city_stats = (
        df.groupby("City")[["Annual_Growth_Rate", "Investment_Score"]]
        .median()
        .to_dict(orient="index")
    )

    st.markdown("---")

    # ---------- Top layout: instructions + dataset stats ----------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📌 How to use")
        st.markdown(
            """
            1. Set the property details in the **left sidebar**  
            2. Click **'🔮 Predict Investment & Future Price'**  
            3. Read the investment decision & 5-year price estimate  
            """
        )

    with col_right:
        st.markdown("### 📈 Dataset Snapshot")
        st.metric("Total Properties", f"{len(df):,}")
        st.metric("Unique Cities", df["City"].nunique())
        st.metric("States Covered", df["State"].nunique())

    # ---------- Sidebar inputs ----------
    with st.sidebar:
        st.title("📥 Property Inputs")

        st.markdown("### 🌍 Location")
        state = st.selectbox("State", sorted(df["State"].unique()))
        city_options = sorted(df[df["State"] == state]["City"].unique())
        city = st.selectbox("City", city_options)
        locality_options = sorted(df[df["City"] == city]["Locality"].unique())
        locality = st.selectbox("Locality", locality_options)

        st.markdown("---")
        st.markdown("### 🏠 Property Details")
        property_type = st.selectbox(
            "Property Type", sorted(df["Property_Type"].unique())
        )
        bhk = st.slider("BHK", min_value=1, max_value=5, value=3, step=1)
        size_sqft = st.slider(
            "Size (SqFt)", min_value=500, max_value=5000, value=2000, step=50
        )
        price_lakhs = st.slider(
            "Current Price (Lakhs)", min_value=10, max_value=500, value=250, step=5
        )
        year_built = st.slider(
            "Year Built",
            min_value=int(df["Year_Built"].min()),
            max_value=int(df["Year_Built"].max()),
            value=2010,
            step=1,
        )
        floor_no = st.slider("Floor No", min_value=0, max_value=30, value=5, step=1)
        total_floors = st.slider(
            "Total Floors in Building", min_value=1, max_value=30, value=15, step=1
        )

        st.markdown("---")
        st.markdown("### 🏫 Neighborhood")
        nearby_schools = st.slider(
            "Nearby Schools (1–10)", min_value=1, max_value=10, value=5, step=1
        )
        nearby_hospitals = st.slider(
            "Nearby Hospitals (1–10)", min_value=1, max_value=10, value=5, step=1
        )

        st.markdown("---")
        st.markdown("### 🧩 Other Attributes")
        furnished_status = st.selectbox(
            "Furnished Status", sorted(df["Furnished_Status"].unique())
        )
        availability_status = st.selectbox(
            "Availability Status", sorted(df["Availability_Status"].unique())
        )
        transport_level = st.selectbox(
            "Public Transport Accessibility",
            sorted(df["Public_Transport_Accessibility"].unique()),
        )
        parking_space = st.selectbox(
            "Parking Space", sorted(df["Parking_Space"].unique())
        )
        security_cat = st.selectbox(
            "Security Level", sorted(df["Security"].unique())
        )
        amenities = st.selectbox(
            "Amenities (pattern)", sorted(df["Amenities"].unique())
        )
        facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
        owner_type = st.selectbox("Owner Type", sorted(df["Owner_Type"].unique()))

        predict_clicked = st.button(
            "🔮 Predict Investment & Future Price", use_container_width=True
        )

    # ---------- Derived / encoded features ----------
    price_per_sqft = price_lakhs / size_sqft
    current_year = 2025
    age_property = current_year - year_built

    furnished_map = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
    furnished_enc = furnished_map.get(furnished_status, 1)

    availability_map = {"Under_Construction": 0, "Ready_to_Move": 1}
    availability_enc = availability_map.get(availability_status, 1)

    level_map = {"Low": 0, "Medium": 1, "High": 2}
    transport_score = level_map.get(transport_level, 1)
    security_score = level_map.get(security_cat, 1)

    city_info = city_stats.get(city, None)
    if city_info is not None:
        annual_growth_rate = city_info["Annual_Growth_Rate"]
        base_investment_score = city_info["Investment_Score"]
    else:
        annual_growth_rate = float(df["Annual_Growth_Rate"].median())
        base_investment_score = float(df["Investment_Score"].median())

    inv_score = base_investment_score
    if price_per_sqft < df[df["City"] == city]["Price_per_SqFt"].median():
        inv_score += 1
    if transport_score == 2:
        inv_score += 1
    if nearby_schools >= 7:
        inv_score += 1
    if nearby_hospitals >= 7:
        inv_score += 1
    inv_score = int(np.clip(inv_score, 0, 5))

    input_dict = {
        # numeric
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": price_per_sqft,
        "Year_Built": year_built,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age_property,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Furnished_Status_Enc": furnished_enc,
        "Availability_Status_Enc": availability_enc,
        "Transport_Score": transport_score,
        "Security_Score": security_score,
        "Annual_Growth_Rate": annual_growth_rate,
        "Investment_Score": inv_score,
        # categorical
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "Furnished_Status": furnished_status,
        "Public_Transport_Accessibility": transport_level,
        "Parking_Space": parking_space,
        "Security": security_cat,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability_status,
    }

    input_df = pd.DataFrame([input_dict], columns=feature_cols)

    with st.expander("📊 Model-ready Input Preview", expanded=False):
        st.dataframe(input_df)

    # ---------- Prediction ----------
    if predict_clicked:
        with st.spinner("Running models..."):
            pred_label, pred_proba, pred_price = make_predictions(
                clf_model, reg_model, input_df
            )

        is_good = bool(pred_label.iloc[0])
        prob_good = float(pred_proba.iloc[0])
        fut_price = float(pred_price.iloc[0])

        st.markdown("---")
        st.subheader("🔮 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Future Price (5 years)",
                f"{fut_price:,.2f} Lakhs",
                help="Regression model prediction for Future_Price_5Y",
            )

        with col2:
            st.metric(
                "Good Investment?",
                "✅ Yes" if is_good else "❌ No",
                help=f"Classification decision. Probability good: {prob_good*100:.1f}%",
            )
            st.progress(min(max(prob_good, 0.0), 1.0))

        st.write(
            f"**Probability that this is a good investment:** `{prob_good*100:.1f}%`"
        )

        st.caption(
            "Note: Models are trained on a synthetic, balanced dataset. "
            "Real-world performance will be weaker and more uncertain."
        )


if __name__ == "__main__":
    main()
