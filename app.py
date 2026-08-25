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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="India Women Safety & Emergency Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title-text">🛡️ India Women Safety & Emergency Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">'
    'Historical women-related crime analysis, future risk prediction and nearby emergency-support facilities.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "cleaned_community_safety_crime_2010_2022.csv"

USER_AGENT = "India-Women-Safety-Dashboard/1.0"

DEFAULT_LAT = 19.0330
DEFAULT_LON = 73.0297

DEFAULT_LOCATION = "Vashi, Navi Mumbai, Maharashtra, India"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset(uploaded_file=None):

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    else:
        try:
            df = pd.read_csv(DATA_FILE)
        except Exception:
            return None

    # Clean column names
    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    # Numeric year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce"
        )

    # Numeric crime columns
    for col in df.columns:
        if col not in ["state", "district", "source_dataset"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["year", "state", "district"]
    )

    df["year"] = df["year"].astype(int)

    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["district"] = df["district"].astype(str).str.upper().str.strip()

    return df


# ============================================================
# FILE UPLOAD
# ============================================================

st.sidebar.header("📂 Crime Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload women-related crime CSV",
    type=["csv"],
    help="Upload the historical women-related crime dataset."
)

df = load_dataset(uploaded_file)


if df is None:

    st.error(
        "Crime dataset not found. "
        "Upload the CSV from the sidebar or keep the CSV file "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# CRIME CATEGORY DEFINITIONS
# ============================================================

CRIME_COLUMNS = {
    "Rape": "rape",

    "Assault on Women": "assault_women",

    "Insult to Modesty": "insult_modesty",

    "Importation of Girls": "importation_girls",

    "Immoral Traffic": "immoral_traffic",

    "Procuration of Minor Girls":
        "procuration_minor_girls",

    "Human Trafficking":
        "human_trafficking",

    "Assault on Women Above 18":
        "assault_women_above18",

    "Assault on Women Below 18":
        "assault_women_below18",

    "Insult to Modesty Above 18":
        "insult_modesty_above18",

    "Insult to Modesty Below 18":
        "insult_modesty_below18",

    "Cyber Explicit Material":
        "cyber_explicit_material",

    "Other Women Cyber Crimes":
        "other_women_cyber_crimes"
}


AVAILABLE_CRIMES = {
    name: column
    for name, column in CRIME_COLUMNS.items()
    if column in df.columns
}


# ============================================================
# LOCATION NORMALIZATION
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).upper()

    value = re.sub(
        r"[^A-Z0-9 ]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371.0

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(
            math.radians(lat1)
        )
        *
        math.cos(
            math.radians(lat2)
        )
        *
        math.sin(dlon / 2) ** 2
    )

    a = max(
        0.0,
        min(1.0, a)
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius * c


# ============================================================
# GEOCODING
# ============================================================

@st.cache_data(ttl=3600)
def geocode_india(query):

    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    params = {
        "q": f"{query}, India",
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1,
        "countrycodes": "in"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers={
                "User-Agent": USER_AGENT
            },
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if not data:
            return None

        result = data[0]

        address = result.get(
            "address",
            {}
        )

        return {
            "lat": float(result["lat"]),
            "lon": float(result["lon"]),
            "display_name":
                result.get(
                    "display_name",
                    query
                ),
            "state":
                address.get("state", ""),
            "district":
                address.get(
                    "state_district",
                    address.get(
                        "district",
                        ""
                    )
                ),
            "city":
                address.get(
                    "city",
                    address.get(
                        "town",
                        address.get(
                            "village",
                            ""
                        )
                    )
                )
        }

    except Exception:
        return None


# ============================================================
# DATASET LOCATION MATCHING
# ============================================================

def find_location_data(
    dataset,
    state_name,
    district_name
):

    state_norm = normalize_text(
        state_name
    )

    district_norm = normalize_text(
        district_name
    )

    data = dataset.copy()

    data["_state_norm"] = (
        data["state"]
        .map(normalize_text)
    )

    data["_district_norm"] = (
        data["district"]
        .map(normalize_text)
    )

    # --------------------------------------------------------
    # Exact district + state match
    # --------------------------------------------------------

    exact = data[
        (
            data["_state_norm"] ==
            state_norm
        )
        &
        (
            data["_district_norm"] ==
            district_norm
        )
    ]

    if len(exact) > 0:

        return (
            exact,
            "District-level historical data"
        )

    # --------------------------------------------------------
    # District-only match
    # --------------------------------------------------------

    district_match = data[
        data["_district_norm"] ==
        district_norm
    ]

    if len(district_match) > 0:

        return (
            district_match,
            "District-level historical data"
        )

    # --------------------------------------------------------
    # State fallback
    # --------------------------------------------------------

    state_match = data[
        data["_state_norm"] ==
        state_norm
    ]

    if len(state_match) > 0:

        return (
            state_match,
            "State-level fallback data"
        )

    return (
        pd.DataFrame(),
        "No matching historical data"
    )


# ============================================================
# SAFETY CLASSIFICATION
# ============================================================

def classify_risk(
    value,
    reference_values
):

    values = pd.Series(
        reference_values
    ).dropna()

    if len(values) < 3:
        return "Moderate"

    low_threshold = values.quantile(
        0.33
    )

    high_threshold = values.quantile(
        0.67
    )

    if value <= low_threshold:
        return "Low"

    elif value >= high_threshold:
        return "High"

    return "Moderate"


def risk_score(
    value,
    reference_values
):

    values = pd.Series(
        reference_values
    ).dropna()

    if len(values) < 3:
        return 50

    percentile = (
        values <= value
    ).mean() * 100

    return round(
        percentile,
        1
    )


# ============================================================
# HISTORICAL CRIME ANALYSIS
# ============================================================

def build_historical_analysis(
    location_df,
    full_df
):

    records = []

    for crime_name, column in AVAILABLE_CRIMES.items():

        yearly = (
            location_df
            .groupby("year")[column]
            .sum()
            .fillna(0)
        )

        if len(yearly) == 0:
            continue

        historical_mean = yearly.mean()

        reference = (
            full_df
            .groupby("year")[column]
            .sum()
            .values
        )

        level = classify_risk(
            historical_mean,
            reference
        )

        score = risk_score(
            historical_mean,
            reference
        )

        records.append({
            "Crime Type": crime_name,
            "Historical Average":
                round(
                    historical_mean,
                    2
                ),
            "Risk Score":
                score,
            "Safety Level":
                level
        })

    return pd.DataFrame(records)


# ============================================================
# FUTURE FORECAST
# ============================================================

def forecast_series(
    yearly_series,
    future_years=5
):

    series = (
        yearly_series
        .dropna()
        .sort_index()
    )

    if len(series) == 0:
        return pd.DataFrame()

    years = np.array(
        series.index,
        dtype=float
    )

    values = np.array(
        series.values,
        dtype=float
    )

    values = np.nan_to_num(
        values,
        nan=0.0
    )

    # --------------------------------------------------------
    # If only one historical year exists
    # --------------------------------------------------------

    if len(values) == 1:

        future_values = np.repeat(
            values[0],
            future_years
        )

    else:

        # Linear trend
        slope, intercept = np.polyfit(
            years,
            values,
            1
        )

        last_year = int(
            years.max()
        )

        future_year_list = np.arange(
            last_year + 1,
            last_year + future_years + 1
        )

        future_values = (
            intercept
            +
            slope * future_year_list
        )

        # Crime count cannot be negative
        future_values = np.maximum(
            future_values,
            0
        )

    future_year_list = np.arange(
        int(years.max()) + 1,
        int(years.max()) + future_years + 1
    )

    historical_df = pd.DataFrame({
        "Year": years.astype(int),
        "Crime Count": values,
        "Type": "Historical"
    })

    forecast_df = pd.DataFrame({
        "Year": future_year_list,
        "Crime Count": future_values,
        "Type": "Predicted"
    })

    return pd.concat(
        [
            historical_df,
            forecast_df
        ],
        ignore_index=True
    )


# ============================================================
# FORECAST SAFETY
# ============================================================

def forecast_risk_level(
    predicted_value,
    historical_values
):

    return classify_risk(
        predicted_value,
        historical_values
    )


# ============================================================
# OVERALL SAFETY
# ============================================================

def calculate_overall_safety(
    location_df,
    full_df
):

    yearly_total = (
        location_df
        .groupby("year")[
            list(AVAILABLE_CRIMES.values())
        ]
        .sum()
        .sum(axis=1)
    )

    reference_total = (
        full_df
        .groupby("year")[
            list(AVAILABLE_CRIMES.values())
        ]
        .sum()
        .sum(axis=1)
    )

    if len(yearly_total) == 0:
        return (
            "Unknown",
            50
        )

    current_average = (
        yearly_total.mean()
    )

    percentile = (
        reference_total <=
        current_average
    ).mean() * 100

    if percentile <= 33:
        level = "Low"

    elif percentile >= 67:
        level = "High"

    else:
        level = "Moderate"

    return (
        level,
        round(percentile, 1)
    )


# ============================================================
# OVERPASS EMERGENCY FACILITIES
# ============================================================

@st.cache_data(ttl=1800)
def get_nearby_facilities(
    lat,
    lon,
    radius_km
):

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

        nwr["social_facility"="shelter"]
            (around:{radius_m},{lat},{lon});

        nwr["name"~"women|woman|mahila|child|children|bal|support|helpline|safety",
            i]
            (around:{radius_m},{lat},{lon});
    );

    out center tags;
    """

    url = (
        "https://overpass-api.de/api/interpreter"
    )

    facilities = []

    try:

        response = requests.post(
            url,
            data={"data": query},
            headers={
                "User-Agent": USER_AGENT
            },
            timeout=25
        )

        if response.status_code != 200:
            return []

        elements = response.json().get(
            "elements",
            []
        )

        for element in elements:

            tags = element.get(
                "tags",
                {}
            )

            e_lat = element.get(
                "lat"
            )

            e_lon = element.get(
                "lon"
            )

            center = element.get(
                "center",
                {}
            )

            if e_lat is None:
                e_lat = center.get(
                    "lat"
                )

            if e_lon is None:
                e_lon = center.get(
                    "lon"
                )

            if e_lat is None or e_lon is None:
                continue

            amenity = tags.get(
                "amenity",
                ""
            ).lower()

            name = (
                tags.get("name")
                or tags.get("official_name")
                or "Emergency Support Facility"
            )

            name_lower = name.lower()

            # ------------------------------------------------
            # Facility classification
            # ------------------------------------------------

            if amenity == "police":

                facility_type = "Police Station"
                color = "red"
                icon = "shield"

            elif amenity in [
                "hospital",
                "clinic"
            ]:

                facility_type = "Hospital / Clinic"
                color = "green"
                icon = "plus"

            elif amenity == "fire_station":

                facility_type = "Fire Station"
                color = "orange"
                icon = "fire"

            elif (
                "women" in name_lower
                or "woman" in name_lower
                or "mahila" in name_lower
                or "sakhi" in name_lower
            ):

                facility_type = "Women Support"
                color = "purple"
                icon = "female"

            elif (
                "child" in name_lower
                or "children" in name_lower
                or "bal " in name_lower
                or "childline" in name_lower
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
                float(e_lon)
            )

            if distance <= radius_km:

                facilities.append({
                    "name": name,
                    "type": facility_type,
                    "lat": float(e_lat),
                    "lon": float(e_lon),
                    "distance_km":
                        round(
                            distance,
                            2