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
# 1. MOCK / EXTENDED DATA GENERATOR & LOADER
# -----------------------------------------------------------------------------
@st.cache_data
def load_safety_data():
    # Base dataset path
    data_path = 'data/raw/cleaned_community_safety_crime_2010_2022.csv'
    
    # Pre-defined locations around Navi Mumbai / Mumbai area for realistic mapping
    locations_db = [
        {"area": "Vashi", "lat": 19.0770, "lon": 72.9986, "risk_score": 75, "risk_level": "High", "past_assaults": 42},
        {"area": "Nerul", "lat": 19.0330, "lon": 73.0169, "risk_score": 45, "risk_level": "Moderate", "past_assaults": 18},
        {"area": "Belapur", "lat": 19.0243, "lon": 73.0488, "risk_score": 25, "risk_level": "Low", "past_assaults": 6},
        {"area": "Kharghar", "lat": 19.0269, "lon": 73.0685, "risk_score": 30, "risk_level": "Low", "past_assaults": 9},
        {"area": "Thane", "lat": 19.2183, "lon": 72.9781, "risk_score": 80, "risk_level": "High", "past_assaults": 55},
        {"area": "Panvel", "lat": 18.9894, "lon": 73.1175, "risk_score": 50, "risk_level": "Moderate", "past_assaults": 22},
    ]
    
    # Essential Support Places per area
    support_places = [
        # Vashi Facilities
        {"area": "Vashi", "name": "Vashi Police Station", "type": "Police Station", "lat": 19.0790, "lon": 73.0000, "icon": "shield-halved", "color": "blue"},
        {"area": "Vashi", "name": "Fortis Hospital Vashi", "type": "Hospital", "lat": 19.0750, "lon": 72.9960, "icon": "hospital", "color": "red"},
        {"area": "Vashi", "name": "Apollo Pharmacy Vashi", "type": "Pharmacy", "lat": 19.0762, "lon": 72.9975, "icon": "pills", "color": "green"},
        {"area": "Vashi", "name": "Vashi Fire Station", "type": "Fire Station", "lat": 19.0740, "lon": 73.0010, "icon": "fire-extinguisher", "color": "orange"},
        {"area": "Vashi", "name": "Women & Child Care Cell", "type": "Women & Child Support", "lat": 19.0785, "lon": 72.9950, "icon": "hand-holding-heart", "color": "purple"},
        {"area": "Vashi", "name": "District Legal Aid Clinic", "type": "Legal Support", "lat": 19.0801, "lon": 72.9990, "icon": "scale-balanced", "color": "darkblue"},

        # Nerul Facilities
        {"area": "Nerul", "name": "Nerul Police Station", "type": "Police Station", "lat": 19.0340, "lon": 73.0180, "icon": "shield-halved", "color": "blue"},
        {"area": "Nerul", "name": "DY Patil Hospital", "type": "Hospital", "lat": 19.0315, "lon": 73.0150, "icon": "hospital", "color": "red"},
        {"area": "Nerul", "name": "Wellness Pharmacy", "type": "Pharmacy", "lat": 19.0335, "lon": 73.0160, "icon": "pills", "color": "green"},
        {"area": "Nerul", "name": "Nerul Fire Station", "type": "Fire Station", "lat": 19.0350, "lon": 73.0195, "icon": "fire-extinguisher", "color": "orange"},
        
        # Kharghar Facilities
        {"area": "Kharghar", "name": "Kharghar Police Station", "type": "Police Station", "lat": 19.0280, "lon": 73.0690, "icon": "shield-halved", "color": "blue"},
        {"area": "Kharghar", "name": "TATA Memorial Hospital", "type": "Hospital", "lat": 19.0250, "lon": 73.0670, "icon": "hospital", "color": "red"},
        {"area": "Kharghar", "name": "Sakhi Women Helpline Desk", "type": "Women & Child Support", "lat": 19.0290, "lon": 73.0700, "icon": "hand-holding-heart", "color": "purple"}
    ]

    df_loc = pd.DataFrame(locations_db)
    df_support = pd.DataFrame(support_places)
    
    return df_loc, df_support

df_loc, df_support = load_safety_data()

# -----------------------------------------------------------------------------
# 2. APPLICATION HEADER & NAVIGATION
# -----------------------------------------------------------------------------
st.title("🛡️ Swaraksha - Smart Safety & Threat Analytics Dashboard")
st.markdown("Real-time safety zone tracking, essential support mapping, and predictive area crime analytics.")

st.sidebar.header("🔍 Location & Threat Controls")

selected_area = st.sidebar.selectbox(
    "Select Target Area / Location:",
    df_loc["area"].unique()
)

radius_km = st.sidebar.slider("Select Radius around Location (in Km):", 1, 10, 3)

# Filter criteria based on user selection
area_info = df_loc[df_loc["area"] == selected_area].iloc[0]
past_crime_count = area_info["past_assaults"]
risk_level = area_info["risk_level"]
risk_score = area_info["risk_score"]

# Future Risk Prediction Model Logic (Simulation based on trend)
predicted_future_assaults = int(past_crime_count * 0.85) if risk_level == "Low" else int(past_crime_count * 1.15)

# -----------------------------------------------------------------------------
# 3. METRICS DASHBOARD & PREDICTION BANNER
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
        st.success(f"Danger Zone: **{risk_level.upper()}**")

with col3:
    st.metric("Recorded Past Assaults", f"{past_crime_count} Cases")

with col4:
    delta_val = predicted_future_assaults - past_crime_count
    st.metric("Predicted Future Risk Trend", f"~{predicted_future_assaults} Cases", delta=f"{delta_val} projected change")

st.divider()

# -----------------------------------------------------------------------------
# 4. MAP INTEGRATION (FOLIUM) - DANGER ZONE & SUPPORT PLACES
# -----------------------------------------------------------------------------
st.subheader(f"📍 Interactive Safety Map for {selected_area} (Radius: {radius_km} km)")

# Set base map position
m = folium.Map(location=[area_info["lat"], area_info["lon"]], zoom_start=14)

# Zone indicator color
zone_color = "red" if risk_level == "High" else "orange" if risk_level == "Moderate" else "green"

# Add Threat Zone Circle Buffer
folium.Circle(
    location=[area_info["lat"], area_info["lon"]],
    radius=radius_km * 1000,
    color=zone_color,
    fill=True,
    fill_color=zone_color,
    fill_opacity=0.2,
    popup=f"{selected_area} Risk Level: {risk_level}"
).add_to(m)

# Add Central Selected Area Marker
folium.Marker(
    location=[area_info["lat"], area_info["lon"]],
    popup=f"<b>Location: {selected_area}</b><br>Threat Level: {risk_level}",
    icon=folium.Icon(color="black", icon="location-pin", prefix="fa")
).add_to(m)

# Add Filtered Nearby Support Places
nearby_places = df_support[df_support["area"] == selected_area]

if not nearby_places.empty:
    for _, row in nearby_places.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"<b>{row['name']}</b><br>Type: {row['type']}",
            tooltip=row["name"],
            icon=folium.Icon(color=row["color"], icon=row["icon"], prefix="fa")
        ).add_to(m)
else:
    # Default fallback marker if area details are sparse
    folium.Marker(
        location=[area_info["lat"] + 0.002, area_info["lon"] + 0.002],
        popup="Emergency Helpline Point",
        icon=folium.Icon(color="blue", icon="shield-halved", prefix="fa")
    ).add_to(m)

# Render Map inside Streamlit UI
st_folium(m, width=1100, height=450)

st.divider()

# -----------------------------------------------------------------------------
# 5. GRAPHS & ANALYTICAL DASHBOARD (RETAINED & EXTENDED)
# -----------------------------------------------------------------------------
st.subheader("📊 Safety Analytics & Historical Trend Graphs")

g_col1, g_col2 = st.columns(2)

with g_col1:
    # Graph 1: Comparative Threat Level by Location
    fig_risk = px.bar(
        df_loc, 
        x='area', 
        y='risk_score',
        color='risk_level',
        color_discrete_map={"High": "#e74c3c", "Moderate": "#f39c12", "Low": "#2ecc71"},
        title="Comparative Danger/Risk Level Across Locations",
        labels={'risk_score': 'Risk Index Score', 'area': 'Location'}
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with g_col2:
    # Graph 2: Past vs Predicted Future Crime Incidents for Selected Area
    comparison_df = pd.DataFrame({
        "Category": ["Past Reported Assaults", "Predicted Future Incidents"],
        "Incidents": [past_crime_count, predicted_future_assaults]
    })
    
    fig_pred = px.pie(
        comparison_df, 
        names="Category", 
        values="Incidents",
        color="Category",
        color_discrete_map={"Past Reported Assaults": "#3498db", "Predicted Future Incidents": "#e74c3c"},
        title=f"Incident Ratio & Future Forecast: {selected_area}",
        hole=0.4
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. SUPPORT FACILITIES LIST & EMERGENCY CONTACTS
# -----------------------------------------------------------------------------
st.subheader("🚑 Nearby Support & Emergency Facilities Breakdown")

if not nearby_places.empty:
    st.dataframe(
        nearby_places[['name', 'type', 'lat', 'lon']].rename(
            columns={'name': 'Facility Name', 'type': 'Category', 'lat': 'Latitude', 'lon': 'Longitude'}
        ),
        use_container_width=True
    )
else:
    st.info("General helpline centers active in radius. Contact 112 for direct dispatch.")