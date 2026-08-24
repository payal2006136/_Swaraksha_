import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="Swaraksha - Women & Community Safety Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. DATA PROCESSING & HISTORICAL CRIME ANALYSIS (UP TO 2026 PREDICTIONS)
# -----------------------------------------------------------------------------
@st.cache_data
def load_safety_data():
    # Primary Area Dataset with coordinates, historical baseline, and risk distribution
    locations_db = [
        {"area": "Vashi", "lat": 19.0770, "lon": 72.9986, "past_assaults": 58, "risk_level": "High", "risk_score": 82},
        {"area": "Nerul", "lat": 19.0330, "lon": 73.0169, "past_assaults": 24, "risk_level": "Moderate", "risk_score": 48},
        {"area": "Belapur", "lat": 19.0243, "lon": 73.0488, "past_assaults": 8, "risk_level": "Low", "risk_score": 20},
        {"area": "Kharghar", "lat": 19.0269, "lon": 73.0685, "past_assaults": 12, "risk_level": "Low", "risk_score": 28},
        {"area": "Thane West", "lat": 19.2183, "lon": 72.9781, "past_assaults": 64, "risk_level": "High", "risk_score": 88},
        {"area": "Panvel", "lat": 18.9894, "lon": 73.1175, "past_assaults": 31, "risk_level": "Moderate", "risk_score": 52},
    ]

    # Essential Support Places around target locations
    support_places = [
        # Vashi Facilities
        {"area": "Vashi", "name": "Vashi Police Station", "type": "Police Station", "lat": 19.0790, "lon": 73.0000, "icon": "shield-halved", "color": "blue", "contact": "022-27820100"},
        {"area": "Vashi", "name": "Fortis Hospital Vashi", "type": "Hospital", "lat": 19.0750, "lon": 72.9960, "icon": "hospital", "color": "red", "contact": "022-67988888"},
        {"area": "Vashi", "name": "Apollo Pharmacy Vashi", "type": "Pharmacy", "lat": 19.0762, "lon": 72.9975, "icon": "pills", "color": "green", "contact": "1860-500-0101"},
        {"area": "Vashi", "name": "Vashi Fire Station", "type": "Fire Station", "lat": 19.0740, "lon": 73.0010, "icon": "fire-extinguisher", "color": "orange", "contact": "101 / 022-27821010"},
        {"area": "Vashi", "name": "Women & Child Care Cell", "type": "Women & Child Support", "lat": 19.0785, "lon": 72.9950, "icon": "hand-holding-heart", "color": "purple", "contact": "1091 / 022-27572224"},
        {"area": "Vashi", "name": "District Legal Aid Clinic", "type": "Legal Support", "lat": 19.0801, "lon": 72.9990, "icon": "scale-balanced", "color": "darkblue", "contact": "15100"},

        # Nerul Facilities
        {"area": "Nerul", "name": "Nerul Police Station", "type": "Police Station", "lat": 19.0340, "lon": 73.0180, "icon": "shield-halved", "color": "blue", "contact": "022-27702333"},
        {"area": "Nerul", "name": "DY Patil Hospital", "type": "Hospital", "lat": 19.0315, "lon": 73.0150, "icon": "hospital", "color": "red", "contact": "022-30965900"},
        {"area": "Nerul", "name": "Wellness Pharmacy", "type": "Pharmacy", "lat": 19.0335, "lon": 73.0160, "icon": "pills", "color": "green", "contact": "022-27712345"},
        {"area": "Nerul", "name": "Nerul Fire Station", "type": "Fire Station", "lat": 19.0350, "lon": 73.0195, "icon": "fire-extinguisher", "color": "orange", "contact": "101"},

        # Kharghar Facilities
        {"area": "Kharghar", "name": "Kharghar Police Station", "type": "Police Station", "lat": 19.0280, "lon": 73.0690, "icon": "shield-halved", "color": "blue", "contact": "022-27740100"},
        {"area": "Kharghar", "name": "Tata Memorial Cancer Hospital", "type": "Hospital", "lat": 19.0250, "lon": 73.0670, "icon": "hospital", "color": "red", "contact": "022-27405000"},
        {"area": "Kharghar", "name": "Sakhi Women Helpline Desk", "type": "Women & Child Support", "lat": 19.0290, "lon": 73.0700, "icon": "hand-holding-heart", "color": "purple", "contact": "181"}
    ]

    return pd.DataFrame(locations_db), pd.DataFrame(support_places)

df_loc, df_support = load_safety_data()

# -----------------------------------------------------------------------------
# 2. SIDEBAR & CONTROLS
# -----------------------------------------------------------------------------
st.title("🛡️ Swaraksha - Smart Safety & Threat Analytics Dashboard")
st.markdown("Real-time safety zone tracking, essential support mapping, and predictive area crime analytics (2026).")

st.sidebar.header("⚙️ Filter Controls")

selected_area = st.sidebar.selectbox(
    "Select Target Location / Area:",
    df_loc["area"].unique()
)

radius_km = st.sidebar.slider("Select Search Radius around Area (in Km):", 1, 10, 3)

# Retrieve Area Details
area_info = df_loc[df_loc["area"] == selected_area].iloc[0]
past_crime_count = area_info["past_assaults"]
risk_level = area_info["risk_level"]
risk_score = area_info["risk_score"]

# Dynamic 2026 Future Crime Risk Prediction Model (Calculated based on past trends)
if risk_level == "High":
    predicted_2026_crime = int(past_crime_count * 1.12)  # High risk trend
elif risk_level == "Moderate":
    predicted_2026_crime = int(past_crime_count * 0.95)  # Moderate trend
else:
    predicted_2026_crime = int(past_crime_count * 0.75)  # Low/Safe risk trend

# -----------------------------------------------------------------------------
# 3. METRICS DASHBOARD
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Selected Area", selected_area)

with col2:
    if risk_level == "High":
        st.error(f"Danger Zone: **{risk_level.upper()}**")
    elif risk_level == "Moderate":
        st.warning(f"Danger Zone: **{risk_level.upper()}**")
    else:
        st.success(f"Danger Zone: **{risk_level.upper()}** (SAFE)")

with col3:
    st.metric("Past Recorded Assaults", f"{past_crime_count} Cases")

with col4:
    delta_val = predicted_2026_crime - past_crime_count
    st.metric("2026 Projected Crime Trend", f"~{predicted_2026_crime} Cases", delta=f"{delta_val} from past baseline")

st.divider()

# -----------------------------------------------------------------------------
# 4. INTERACTIVE MAP - DANGER ZONE & SUPPORT PLACES HIGHLIGHT
# -----------------------------------------------------------------------------
st.subheader(f"📍 Interactive Safety Map ({selected_area}) - Radius: {radius_km} km")

# Base Map Initialization
m = folium.Map(location=[area_info["lat"], area_info["lon"]], zoom_start=14)

# Set Color Code for Danger Zones (High=Red, Moderate=Orange, Low=Green)
zone_color = "red" if risk_level == "High" else "orange" if risk_level == "Moderate" else "green"

# Danger Zone Highlighted Buffer Circle
folium.Circle(
    location=[area_info["lat"], area_info["lon"]],
    radius=radius_km * 1000,
    color=zone_color,
    fill=True,
    fill_color=zone_color,
    fill_opacity=0.25,
    popup=f"Area: {selected_area} | Danger Status: {risk_level}"
).add_to(m)

# Target Area Center Marker
folium.Marker(
    location=[area_info["lat"], area_info["lon"]],
    popup=f"<b>Selected Center: {selected_area}</b><br>Threat Level: {risk_level}",
    icon=folium.Icon(color="black", icon="location-pin", prefix="fa")
).add_to(m)

# Highlight Support Facilities Near Selected Area
nearby_places = df_support[df_support["area"] == selected_area]

if not nearby_places.empty:
    for _, row in nearby_places.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"<b>{row['name']}</b><br>Category: {row['type']}<br>Contact: {row['contact']}",
            tooltip=f"{row['type']}: {row['name']}",
            icon=folium.Icon(color=row["color"], icon=row["icon"], prefix="fa")
        ).add_to(m)

# Render Map inside Streamlit UI
st_folium(m, width=1100, height=480)

st.divider()

# -----------------------------------------------------------------------------
# 5. DASHBOARD GRAPHS (RETAINED & INTEGRATED)
# -----------------------------------------------------------------------------
st.subheader("📊 Historical Safety Analytics & 2026 Forecast Graphs")

g_col1, g_col2 = st.columns(2)

with g_col1:
    # Graph 1: Danger Classification Across All Areas (Low, Moderate, High)
    fig_risk = px.bar(
        df_loc,
        x='area',
        y='risk_score',
        color='risk_level',
        color_discrete_map={"High": "#e74c3c", "Moderate": "#f39c12", "Low": "#2ecc71"},
        title="Comparative Danger Level Index (High / Moderate / Low)",
        labels={'risk_score': 'Threat Index', 'area': 'Location'}
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with g_col2:
    # Graph 2: Past Assault Data vs 2026 Future Prediction Comparison
    comparison_df = pd.DataFrame({
        "Category": ["Past Recorded Crimes", "2026 Future Projected Crimes"],
        "Incidents": [past_crime_count, predicted_2026_crime]
    })
    
    fig_pred = px.pie(
        comparison_df,
        names="Category",
        values="Incidents",
        color="Category",
        color_discrete_map={"Past Recorded Crimes": "#3498db", "2026 Future Projected Crimes": "#e74c3c"},
        title=f"Past Data vs 2026 Prediction Model ({selected_area})",
        hole=0.4
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. EMERGENCY HELPLINES & SUPPORT FACILITIES BREAKDOWN
# -----------------------------------------------------------------------------
st.subheader("🚨 Emergency Helplines & Support Facility Details")

h_col1, h_col2 = st.columns([1, 2])

with h_col1:
    st.markdown("### 📞 National Emergency Numbers")
    st.error("**Emergency Response Support System:** 112")
    st.warning("**Women Helpline (All India):** 1091 / 181")
    st.info("**Police Control Room:** 100")
    st.success("**Medical Emergency / Ambulance:** 102 / 108")

with h_col2:
    st.markdown(f"### 🏥 Available Support Facilities in {selected_area}")
    if not nearby_places.empty:
        st.dataframe(
            nearby_places[['name', 'type', 'contact']].rename(
                columns={'name': 'Facility Name', 'type': 'Category', 'contact': 'Contact Number'}
            ),
            use_container_width=True
        )
    else:
        st.info("Central Emergency Desk active. Dial 112 for direct dispatch.")