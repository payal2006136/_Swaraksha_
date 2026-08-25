import math
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium

from folium.plugins import Fullscreen
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="India Women Safety & Emergency Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }

    .title-text {
        font-size: 32px;
        font-weight: 900;
        color: #38bdf8 !important;
        margin-bottom: 5px;
    }

    .sub-text {
        color: #94a3b8 !important;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #f3f4f6 !important;
    }

    div[data-testid="metric-container"] {
        background-color: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        padding: 15px !important;
        border-radius: 12px !important;
    }

    h1, h2, h3, h4, p, label {
        color: #f8fafc !important;
    }

    .safety-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #1e293b;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title-text">🛡️ India Women Safety & Emergency Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-text">'
    "Historical women-related crime analysis, future risk prediction "
    "and nearby emergency-support facilities."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "cleaned_community_safety_crime_2010_2022.csv"
USER_AGENT = "India-Women-Safety-Dashboard/1.0"

DEFAULT_LAT = 19.0330
DEFAULT_LON = 73.0297


# ============================================================
# LOAD DATASET FROM SAME GITHUB FOLDER
# ============================================================

@st.cache_data
def load_dataset():
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(
            f"Dataset not found: {DATA_FILE}. "
            "Make sure the CSV is in the same GitHub folder as app.py."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        st.stop()

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    if "year" not in data.columns:
        st.error("The dataset must contain a 'year' column.")
        st.stop()

    if "state" not in data.columns or "district" not in data.columns:
        st.error("The dataset must contain 'state' and 'district' columns.")
        st.stop()

    data["year"] = pd.to_numeric(
        data["year"], errors="coerce"
    )

    for column in data.columns:
        if column not in ["state", "district", "source_dataset"]:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    data = data.dropna(
        subset=["year", "state", "district"]
    ).copy()

    data["year"] = data["year"].astype(int)
    data["state"] = data["state"].astype(str).str.upper().str.strip()
    data["district"] = data["district"].astype(str).str.upper().str.strip()

    return data


df = load_dataset()


# ============================================================
# WOMEN-RELATED CRIME CATEGORIES
# ============================================================

CRIME_COLUMNS = {
    "Rape": "rape",
    "Assault on Women": "assault_women",
    "Insult to Modesty": "insult_modesty",
    "Importation of Girls": "importation_girls",
    "Immoral Traffic": "immoral_traffic",
    "Procuration of Minor Girls": "procuration_minor_girls",
    "Human Trafficking": "human_trafficking",
    "Assault on Women Above 18": "assault_women_above18",
    "Assault on Women Below 18": "assault_women_below18",
    "Insult to Modesty Above 18": "insult_modesty_above18",
    "Insult to Modesty Below 18": "insult_modesty_below18",
    "Cyber Explicit Material": "cyber_explicit_material",
    "Other Women Cyber Crimes": "other_women_cyber_crimes",
}

AVAILABLE_CRIMES = {
    name: column
    for name, column in CRIME_COLUMNS.items()
    if column in df.columns
}


if not AVAILABLE_CRIMES:
    st.error(
        "No recognized women-related crime columns were found in the CSV."
    )
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).upper()
    value = re.sub(r"[^A-Z0-9 ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


# ============================================================
# GEOCODING - ANY INDIAN LOCATION
# ============================================================

@st.cache_data(ttl=3600)
def geocode_india(query):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": f"{query}, India",
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1,
        "countrycodes": "in",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )

        if response.status_code != 200:
            return None

        results = response.json()

        if not results:
            return None

        result = results[0]
        address = result.get("address", {})

        return {
            "lat": float(result["lat"]),
            "lon": float(result["lon"]),
            "display_name": result.get("display_name", query),
            "state": address.get("state", ""),
            "district": address.get(
                "state_district",
                address.get("district", ""),
            ),
            "city": address.get(
                "city",
                address.get(
                    "town",
                    address.get("village", ""),
                ),
            ),
        }

    except Exception:
        return None


# ============================================================
# MATCH LOCATION TO HISTORICAL DATA
# ============================================================

def find_location_data(dataset, state_name, district_name):
    state_norm = normalize_text(state_name)
    district_norm = normalize_text(district_name)

    data = dataset.copy()

    data["_state_norm"] = data["state"].map(normalize_text)
    data["_district_norm"] = data["district"].map(normalize_text)

    exact = data[
        (data["_state_norm"] == state_norm)
        & (data["_district_norm"] == district_norm)
    ]

    if len(exact) > 0:
        return exact, "District-level historical data"

    district_match = data[
        data["_district_norm"] == district_norm
    ]

    if len(district_match) > 0:
        return district_match, "District-level historical data"

    state_match = data[
        data["_state_norm"] == state_norm
    ]

    if len(state_match) > 0:
        return state_match, "State-level fallback data"

    return pd.DataFrame(), "No matching historical data"


# ============================================================
# LOW / MODERATE / HIGH CLASSIFICATION
# ============================================================

def classify_risk(value, reference_values):
    values = pd.Series(reference_values).dropna()

    if len(values) < 3:
        return "Moderate"

    low_threshold = values.quantile(0.33)
    high_threshold = values.quantile(0.67)

    if value <= low_threshold:
        return "Low"

    if value >= high_threshold:
        return "High"

    return "Moderate"


def risk_score(value, reference_values):
    values = pd.Series(reference_values).dropna()

    if len(values) < 3:
        return 50.0

    percentile = (values <= value).mean() * 100
    return round(percentile, 1)


# ============================================================
# HISTORICAL SAFETY ANALYSIS
# ============================================================

def build_historical_analysis(location_df, full_df):
    records = []

    for crime_name, column in AVAILABLE_CRIMES.items():
        yearly = (
            location_df.groupby("year")[column]
            .sum()
            .fillna(0)
        )

        if len(yearly) == 0:
            continue

        historical_mean = yearly.mean()

        reference = (
            full_df.groupby("year")[column]
            .sum()
            .values
        )

        records.append(
            {
                "Crime Type": crime_name,
                "Historical Average": round(historical_mean, 2),
                "Risk Score": risk_score(
                    historical_mean, reference
                ),
                "Safety Level": classify_risk(
                    historical_mean, reference
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# FUTURE FORECAST
# ============================================================

def forecast_series(yearly_series, future_years=5):
    series = yearly_series.dropna().sort_index()

    if len(series) == 0:
        return pd.DataFrame()

    years = np.asarray(series.index, dtype=float)
    values = np.asarray(series.values, dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    last_year = int(years.max())

    if len(values) == 1:
        future_values = np.repeat(
            values[0],
            future_years,
        )
    else:
        slope, intercept = np.polyfit(
            years,
            values,
            1,
        )

        future_years_array = np.arange(
            last_year + 1,
            last_year + future_years + 1,
        )

        future_values = intercept + slope * future_years_array
        future_values = np.maximum(future_values, 0)

    future_year_list = np.arange(
        last_year + 1,
        last_year + future_years + 1,
    )

    historical_df = pd.DataFrame(
        {
            "Year": years.astype(int),
            "Crime Count": values,
            "Type": "Historical",
        }
    )

    forecast_df = pd.DataFrame(
        {
            "Year": future_year_list,
            "Crime Count": future_values,
            "Type": "Predicted",
        }
    )

    return pd.concat(
        [historical_df, forecast_df],
        ignore_index=True,
    )


# ============================================================
# OVERALL SAFETY
# ============================================================

def calculate_overall_safety(location_df, full_df):
    crime_columns = list(AVAILABLE_CRIMES.values())

    yearly_total = (
        location_df.groupby("year")[crime_columns]
        .sum()
        .sum(axis=1)
    )

    reference_total = (
        full_df.groupby("year")[crime_columns]
        .sum()
        .sum(axis=1)
    )

    if len(yearly_total) == 0:
        return "Unknown", 50.0

    current_average = yearly_total.mean()

    percentile = (
        (reference_total <= current_average).mean()
        * 100
    )

    if percentile <= 33:
        level = "Low"
    elif percentile >= 67:
        level = "High"
    else:
        level = "Moderate"

    return level, round(percentile, 1)


# ============================================================
# NEARBY EMERGENCY / SUPPORT FACILITIES
# ============================================================

@st.cache_data(ttl=1800)
def get_nearby_facilities(lat, lon, radius_km):
    radius_m = radius_km * 1000

    query = f"""
    [out:json][timeout:20];

    (
        nwr["amenity"="police"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="hospital"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="clinic"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="fire_station"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="social_facility"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="shelter"]
            (around:{radius_m},{lat},{lon});

        nwr["name"~"women|woman|mahila|sakhi|child|children|bal|support|helpline|safety",i]
            (around:{radius_m},{lat},{lon});
    );

    out center tags;
    """

    url = "https://overpass-api.de/api/interpreter"

    facilities = []

    try:
        response = requests.post(
            url,
            data={"data": query},
            headers={"User-Agent": USER_AGENT},
            timeout=25,
        )

        if response.status_code != 200:
            return []

        elements = response.json().get("elements", [])

        for element in elements:
            tags = element.get("tags", {})

            e_lat = element.get("lat")
            e_lon = element.get("lon")

            center = element.get("center", {})

            if e_lat is None:
                e_lat = center.get("lat")

            if e_lon is None:
                e_lon = center.get("lon")

            if e_lat is None or e_lon is None:
                continue

            amenity = tags.get("amenity", "").lower()

            name = (
                tags.get("name")
                or tags.get("official_name")
                or "Emergency Support Facility"
            )

            name_lower = name.lower()

            if amenity == "police":
                facility_type = "Police Station"
                color = "red"
                icon = "shield"

            elif amenity in ["hospital", "clinic"]:
                facility_type = "Hospital / Clinic"
                color = "green"
                icon = "plus"

            elif amenity == "fire_station":
                facility_type = "Fire Station"
                color = "orange"
                icon = "fire"

            elif any(
                word in name_lower
                for word in [
                    "women",
                    "woman",
                    "mahila",
                    "sakhi",
                ]
            ):
                facility_type = "Women Support"
                color = "purple"
                icon = "female"

            elif any(
                word in name_lower
                for word in [
                    "child",
                    "children",
                    "bal ",
                    "childline",
                ]
            ):
                facility_type = "Child Support"
                color = "pink"
                icon = "child"

            else:
                facility_type = "Support Center"
                color = "cadetblue"
                icon = "info-sign"

            distance = haversine_km(
                lat,
                lon,
                float(e_lat),
                float(e_lon),
            )

            if distance <= radius_km:
                facilities.append(
                    {
                        "name": name,
                        "type": facility_type,
                        "lat": float(e_lat),
                        "lon": float(e_lon),
                        "distance_km": round(distance, 2),
                        "phone": tags.get(
                            "phone",
                            tags.get(
                                "contact:phone",
                                "N/A",
                            ),
                        ),
                        "color": color,
                        "icon": icon,
                    }
                )

    except Exception:
        return []

    unique = {}

    for facility in facilities:
        key = (
            normalize_text(facility["name"]),
            facility["type"],
        )

        if key not in unique:
            unique[key] = facility

    return sorted(
        unique.values(),
        key=lambda item: item["distance_km"],
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📍 Location Settings")

search_query = st.sidebar.text_input(
    "Enter any place in India:",
    value="Vashi, Navi Mumbai",
)

radius_km = st.sidebar.slider(
    "Safety & Emergency Radius (km):",
    min_value=1,
    max_value=25,
    value=10,
)

selected_crime = st.sidebar.selectbox(
    "Crime type for detailed analysis:",
    list(AVAILABLE_CRIMES.keys()),
)

forecast_years = st.sidebar.slider(
    "Future prediction years:",
    min_value=1,
    max_value=5,
    value=5,
)


# ============================================================
# SESSION LOCATION
# ============================================================

if "location" not in st.session_state:
    st.session_state.location = {
        "lat": DEFAULT_LAT,
        "lon": DEFAULT_LON,
        "display_name": "Vashi, Navi Mumbai, Maharashtra, India",
        "state": "Maharashtra",
        "district": "Thane",
        "city": "Vashi",
    }


# ============================================================
# SEARCH BUTTON
# ============================================================

if st.sidebar.button(
    "🔎 Analyze Selected Area",
    use_container_width=True,
):
    location = geocode_india(search_query)

    if location is not None:
        st.session_state.location = location
        st.sidebar.success("Location found and analyzed!")
    else:
        st.sidebar.error(
            "Location not found. Try including the city and state."
        )


location = st.session_state.location

cur_lat = location["lat"]
cur_lon = location["lon"]

state_name = location["state"]
district_name = location["district"]


# ============================================================
# CRIME DATA MATCH
# ============================================================

location_df, match_type = find_location_data(
    df,
    state_name,
    district_name,
)


# ============================================================
# CRIME DASHBOARD
# ============================================================

if len(location_df) == 0:
    st.warning(
        f"Historical crime data was not found for "
        f"**{location.get('display_name', search_query)}**. "
        "Emergency facilities and the map can still be displayed, "
        "but a crime-based safety score cannot be calculated."
    )

else:
    overall_level, overall_score = calculate_overall_safety(
        location_df,
        df,
    )

    st.info(f"📊 Analysis level: **{match_type}**")

    latest_year = int(location_df["year"].max())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📍 Selected Area",
        location.get("city", search_query).title(),
    )

    col2.metric(
        "🛡️ Overall Safety",
        overall_level,
    )

    col3.metric(
        "📊 Risk Score",
        f"{overall_score}/100",
    )

    col4.metric(
        "📅 Latest Dataset Year",
        latest_year,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CRIME SAFETY LEVELS
    # --------------------------------------------------------

    st.subheader("🚦 Women-Related Crime Safety Levels")

    safety_df = build_historical_analysis(
        location_df,
        df,
    )

    if len(safety_df) > 0:
        st.dataframe(
            safety_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Safety levels are statistical classifications based on "
        "historical crime values in the available dataset. "
        "They are not a guarantee of personal safety."
    )

    # --------------------------------------------------------
    # SELECTED CRIME HISTORY
    # --------------------------------------------------------

    selected_column = AVAILABLE_CRIMES[selected_crime]

    yearly_crime = (
        location_df.groupby("year")[selected_column]
        .sum()
        .sort_index()
    )

    if len(yearly_crime) > 0:
        st.subheader(
            f"📈 Historical Analysis — {selected_crime}"
        )

        historical_chart = pd.DataFrame(
            {
                "Year": yearly_crime.index,
                "Crime Count": yearly_crime.values,
            }
        )

        st.line_chart(
            historical_chart.set_index("Year")
        )

    # --------------------------------------------------------
    # CATEGORY COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "📊 Women-Related Crime Category Comparison"
    )

    category_totals = []

    for crime_name, column in AVAILABLE_CRIMES.items():
        total = location_df[column].fillna(0).sum()

        category_totals.append(
            {
                "Crime Type": crime_name,
                "Total Cases": round(total, 2),
            }
        )

    category_df = pd.DataFrame(
        category_totals
    ).sort_values(
        "Total Cases",
        ascending=False,
    )

    st.bar_chart(
        category_df.set_index("Crime Type")
    )

    # --------------------------------------------------------
    # FUTURE PREDICTION
    # --------------------------------------------------------

    st.subheader(
        f"🔮 Future Prediction — {selected_crime}"
    )

    forecast_df = forecast_series(
        yearly_crime,
        future_years=forecast_years,
    )

    if len(forecast_df) > 0:
        chart_df = forecast_df.pivot(
            index="Year",
            columns="Type",
            values="Crime Count",
        )

        st.line_chart(chart_df)

        prediction_rows = []

        historical_values = yearly_crime.values

        future_only = forecast_df[
            forecast_df["Type"] == "Predicted"
        ]

        for _, row in future_only.iterrows():
            predicted = row["Crime Count"]

            prediction_rows.append(
                {
                    "Year": int(row["Year"]),
                    "Predicted Cases": round(predicted, 2),
                    "Predicted Safety Level": classify_risk(
                        predicted,
                        historical_values,
                    ),
                }
            )

        prediction_table = pd.DataFrame(
            prediction_rows
        )

        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True,
        )

        first_prediction = future_only.iloc[0]["Crime Count"]
        last_prediction = future_only.iloc[-1]["Crime Count"]

        if last_prediction > first_prediction * 1.05:
            st.warning(
                f"⚠️ Historical trend indicates a possible increase "
                f"in {selected_crime} during the forecast period."
            )
        elif last_prediction < first_prediction * 0.95:
            st.success(
                f"✅ Historical trend indicates a possible decrease "
                f"in {selected_crime} during the forecast period."
            )
        else:
            st.info(
                f"ℹ️ Historical trend indicates a relatively stable "
                f"pattern for {selected_crime}."
            )

    # --------------------------------------------------------
    # OVERALL CRIME TREND
    # --------------------------------------------------------

    st.subheader("📉 Overall Women-Related Crime Trend")

    all_crime_yearly = (
        location_df.groupby("year")[list(AVAILABLE_CRIMES.values())]
        .sum()
        .sum(axis=1)
    )

    overall_chart = pd.DataFrame(
        {
            "Year": all_crime_yearly.index,
            "Total Women-Related Crime": all_crime_yearly.values,
        }
    )

    st.line_chart(
        overall_chart.set_index("Year")
    )


# ============================================================
# EMERGENCY FACILITY MAP
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("🗺️ Emergency & Women-Safety Support Map")

st.caption(
    f"Facilities within approximately {radius_km} km "
    "of the selected location."
)

facilities = get_nearby_facilities(
    cur_lat,
    cur_lon,
    radius_km,
)


# ============================================================
# FACILITY COUNTS
# ============================================================

facility_counts = {
    "Police Station": 0,
    "Hospital / Clinic": 0,
    "Fire Station": 0,
    "Women Support": 0,
    "Child Support": 0,
    "Support Center": 0,
}

for facility in facilities:
    facility_type = facility["type"]

    if facility_type in facility_counts:
        facility_counts[facility_type] += 1


f1, f2, f3, f4, f5 = st.columns(5)

f1.metric(
    "👮 Police",
    facility_counts["Police Station"],
)

f2.metric(
    "🏥 Hospital",
    facility_counts["Hospital / Clinic"],
)

f3.metric(
    "🚒 Fire",
    facility_counts["Fire Station"],
)

f4.metric(
    "👩 Women Support",
    facility_counts["Women Support"],
)

f5.metric(
    "🧒 Child Support",
    facility_counts["Child Support"],
)


# ============================================================
# MAP
# ============================================================

m = folium.Map(
    location=[cur_lat, cur_lon],
    zoom_start=13,
    tiles="CartoDB dark_matter",
)

Fullscreen(
    position="topright"
).add_to(m)


# Selected area

folium.Marker(
    [cur_lat, cur_lon],
    popup=(
        "<b>Selected Location</b><br>"
        + str(
            location.get(
                "display_name",
                search_query,
            )
        )
    ),
    tooltip="📍 Selected Area",
    icon=folium.Icon(
        color="blue",
        icon="user",
        prefix="fa",
    ),
).add_to(m)


# Search radius

folium.Circle(
    [cur_lat, cur_lon],
    radius=radius_km * 1000,
    color="#38bdf8",
    fill=True,
    fill_opacity=0.12,
    popup=f"Emergency search radius: {radius_km} km",
).add_to(m)


# Facility markers

for facility in facilities:
    popup_html = f"""
    <b>{facility['name']}</b><br>
    Category: {facility['type']}<br>
    Distance: {facility['distance_km']} km<br>
    Phone: {facility['phone']}
    """

    folium.Marker(
        [facility["lat"], facility["lon"]],
        popup=popup_html,
        tooltip=(
            f"{facility['type']} — "
            f"{facility['name']}"
        ),
        icon=folium.Icon(
            color=facility["color"],
            icon=facility["icon"],
            prefix="fa",
        ),
    ).add_to(m)


st_folium(
    m,
    width="100%",
    height=600,
)


# ============================================================
# FACILITY DIRECTORY
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.subheader(
    "📋 Nearby Emergency & Support Directory"
)

if facilities:
    facilities_df = pd.DataFrame(
        facilities
    )[
        [
            "name",
            "type",
            "distance_km",
            "phone",
        ]
    ]

    facilities_df.columns = [
        "Facility Name",
        "Category",
        "Distance (km)",
        "Contact Number",
    ]

    st.dataframe(
        facilities_df,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning(
        "No mapped emergency/support facilities "
        "were found within the selected radius."
    )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("ℹ️ How This Dashboard Works")

st.markdown(
    """
    **Historical Crime Analysis:** The dashboard studies the
    women-related crime records available in the CSV dataset.

    **Safety Classification:** Each crime category is separately
    classified as Low, Moderate or High using historical values.
    Therefore, the result can be mixed rather than automatically
    marking every crime category as High.

    **Future Prediction:** Historical yearly values are used to
    estimate the trend for the selected future period.

    **Location Matching:** The dashboard first attempts to use
    district-level historical data. If the district is unavailable,
    it falls back to state-level historical data instead of
    inventing a district result.

    **Emergency Map:** Nearby police stations, hospitals/clinics,
    fire stations and mapped women/child/support facilities are
    displayed around the selected location.

    **Important:** Future predictions are statistical estimates
    based on historical patterns and should not be treated as a
    guarantee of future crime or personal safety.
    """
)
