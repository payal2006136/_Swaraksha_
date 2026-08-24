# ============================================================
# SWARAKSHA
# Know the Pattern. Find the Help.
# India-focused Community Safety + GIS + GPS + Data Science
# ============================================================

import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium

from folium.plugins import Fullscreen, LocateControl
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Swaraksha | Know the Pattern. Find the Help.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #07111f;
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: #0d1726;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f2742, #102b38);
        border: 1px solid #29445e;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
    }

    .card {
        padding: 18px;
        border-radius: 15px;
        background: #111c2c;
        border: 1px solid #26364d;
        margin-bottom: 12px;
    }

    .warning {
        padding: 15px;
        border-radius: 12px;
        background: #241f0b;
        border: 1px solid #76621c;
        color: #f8fafc;
    }

    .low {
        padding: 18px;
        border-radius: 15px;
        background: #0d2b20;
        border: 2px solid #22c55e;
        text-align: center;
    }

    .moderate {
        padding: 18px;
        border-radius: 15px;
        background: #30270d;
        border: 2px solid #eab308;
        text-align: center;
    }

    .high {
        padding: 18px;
        border-radius: 15px;
        background: #321416;
        border: 2px solid #ef4444;
        text-align: center;
    }

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

INDIA_LAT = 22.5937
INDIA_LON = 78.9629

USER_AGENT = "Swaraksha-Safety-App/1.0"

DATA_CANDIDATES = [
    Path("data/raw/cleaned_community_safety_crime_dataset.csv"),
    Path("data/cleaned_community_safety_crime_dataset.csv"),
    Path("cleaned_community_safety_crime_dataset.csv"),
    Path("data.csv"),
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column(name):
    """Normalize column names for easier automatic detection."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df, keywords):
    """
    Automatically locate a column using possible keywords.
    Returns the first suitable column or None.
    """
    normalized = {
        normalize_column(col): col
        for col in df.columns
    }

    for key in keywords:
        key_norm = normalize_column(key)

        for norm_col, original in normalized.items():
            if key_norm == norm_col:
                return original

        for norm_col, original in normalized.items():
            if key_norm in norm_col:
                return original

    return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two coordinates in kilometres."""
    earth_radius = 6371.0

    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))

    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2 * earth_radius * math.asin(math.sqrt(a))


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset():
    """Load the user's crime dataset."""

    selected_file = None

    for path in DATA_CANDIDATES:
        if path.exists():
            selected_file = path
            break

    if selected_file is None:
        return None, None

    try:
        df = pd.read_csv(selected_file)

        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        return df, str(selected_file)

    except Exception as exc:
        return None, f"Dataset loading error: {exc}"


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(df):

    state_col = find_column(
        df,
        [
            "state",
            "stateut",
            "stateutname",
            "states",
        ],
    )

    district_col = find_column(
        df,
        [
            "district",
            "districtname",
        ],
    )

    year_col = find_column(
        df,
        [
            "year",
            "yr",
        ],
    )

    women_columns = []
    child_columns = []
    sexual_columns = []
    rape_columns = []
    pocso_columns = []
    crime_columns = []

    for col in df.columns:

        name = normalize_column(col)

        if any(
            word in name
            for word in [
                "women",
                "woman",
                "female",
                "dowry",
                "molestation",
                "stalking",
                "harassment",
                "cruelty",
                "kidnappingwomen",
            ]
        ):
            women_columns.append(col)

        if any(
            word in name
            for word in [
                "child",
                "children",
                "minor",
                "pocso",
            ]
        ):
            child_columns.append(col)

        if any(
            word in name
            for word in [
                "sexual",
                "assault",
                "molestation",
                "harassment",
            ]
        ):
            sexual_columns.append(col)

        if "rape" in name:
            rape_columns.append(col)

        if "pocso" in name:
            pocso_columns.append(col)

        if any(
            word in name
            for word in [
                "crime",
                "rape",
                "assault",
                "kidnap",
                "abduction",
                "harassment",
                "violence",
                "murder",
                "theft",
                "sexual",
                "pocso",
            ]
        ):
            crime_columns.append(col)

    return {
        "state": state_col,
        "district": district_col,
        "year": year_col,
        "women": list(dict.fromkeys(women_columns)),
        "child": list(dict.fromkeys(child_columns)),
        "sexual": list(dict.fromkeys(sexual_columns)),
        "rape": list(dict.fromkeys(rape_columns)),
        "pocso": list(dict.fromkeys(pocso_columns)),
        "crime": list(dict.fromkeys(crime_columns)),
    }


# ============================================================
# LOCATION / DATASET FILTERING
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def filter_location_data(
    df,
    columns,
    state=None,
    district=None,
):

    result = df.copy()

    state_col = columns["state"]
    district_col = columns["district"]

    if state_col and state:
        result = result[
            result[state_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == str(state).strip().lower()
        ]

    if district_col and district:
        result = result[
            result[district_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == str(district).strip().lower()
        ]

    return result


# ============================================================
# CRIME SCORE
# ============================================================

def calculate_indicator(df, columns):
    """
    Calculates an explainable historical pattern indicator.

    IMPORTANT:
    This is NOT a real-world safety guarantee.
    """

    if df is None or df.empty:
        return {
            "score": 0.0,
            "level": "🟢 LOW",
            "description": "No matching historical records were found.",
            "factors": [],
        }

    relevant_columns = list(
        dict.fromkeys(
            columns["women"]
            + columns["child"]
            + columns["sexual"]
            + columns["rape"]
            + columns["pocso"]
        )
    )

    if not relevant_columns:
        relevant_columns = columns["crime"]

    if not relevant_columns:
        return {
            "score": 0.0,
            "level": "🟢 LOW",
            "description": "No relevant numeric crime variables were detected.",
            "factors": [],
        }

    values = {}

    for col in relevant_columns:
        if col in df.columns:
            values[col] = safe_numeric(df[col]).sum()

    if not values:
        return {
            "score": 0.0,
            "level": "🟢 LOW",
            "description": "Relevant numeric crime data was not available.",
            "factors": [],
        }

    total = sum(values.values())

    if total <= 0:
        score = 0.0
    else:
        positive = [
            value
            for value in values.values()
            if value > 0
        ]

        if positive:
            mean_value = float(np.mean(positive))
            score = mean_value
        else:
            score = 0.0

    # Convert the raw score to a relative 0–2 indicator.
    positive_values = [
        value
        for value in values.values()
        if value > 0
    ]

    if positive_values:
        median_value = float(
            np.median(positive_values)
        )

        if median_value > 0:
            relative_score = score / median_value
        else:
            relative_score = 0.0
    else:
        relative_score = 0.0

    # --------------------------------------------------------
    # FIXED INDENTATION BLOCK
    # --------------------------------------------------------

    if relative_score < 0.67:
        level = "🟢 LOW"
    elif relative_score < 1.33:
        level = "🟡 MODERATE"
    else:
        level = "🔴 HIGH"

    ranked = sorted(
        values.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    factors = [
        (str(col), float(value))
        for col, value in ranked[:5]
        if value > 0
    ]

    return {
        "score": round(relative_score, 2),
        "level": level,
        "description": (
            "Historical reported-crime pattern indicator "
            "based on available crime variables."
        ),
        "factors": factors,
    }


# ============================================================
# YEARLY ANALYSIS
# ============================================================

def yearly_analysis(df, columns):

    year_col = columns["year"]

    if year_col is None or df.empty:
        return pd.DataFrame()

    relevant = list(
        dict.fromkeys(
            columns["women"]
            + columns["child"]
            + columns["sexual"]
            + columns["rape"]
            + columns["pocso"]
        )
    )

    if not relevant:
        relevant = columns["crime"]

    if not relevant:
        return pd.DataFrame()

    temp = df.copy()

    temp["_year"] = pd.to_numeric(
        temp[year_col],
        errors="coerce",
    )

    temp = temp.dropna(
        subset=["_year"]
    )

    for col in relevant:
        temp[col] = safe_numeric(temp[col])

    temp["Total_Selected_Crime"] = temp[relevant].sum(axis=1)

    grouped = (
        temp.groupby("_year")["Total_Selected_Crime"]
        .sum()
        .reset_index()
        .sort_values("_year")
    )

    grouped["_year"] = grouped["_year"].astype(int)

    return grouped


# ============================================================
# SIMPLE TIME-AWARE FUTURE ESTIMATE
# ============================================================

def future_projection(df, columns):

    yearly = yearly_analysis(
        df,
        columns,
    )

    if yearly.empty or len(yearly) < 2:
        return pd.DataFrame()

    x = yearly["_year"].values.astype(float)
    y = yearly["Total_Selected_Crime"].values.astype(float)

    if len(x) < 2:
        return pd.DataFrame()

    try:
        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )
    except Exception:
        return pd.DataFrame()

    last_year = int(max(x))

    future_years = np.arange(
        last_year + 1,
        last_year + 4,
    )

    predicted = (
        slope * future_years
        + intercept
    )

    predicted = np.maximum(
        predicted,
        0,
    )

    return pd.DataFrame(
        {
            "Year": future_years,
            "Estimated reported cases": np.round(
                predicted,
                0,
            ).astype(int),
        }
    )


# ============================================================
# GEOCODING
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_india(query):

    if not query:
        return None

    params = {
        "q": f"{query}, India",
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "in",
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            return None

        item = data[0]

        return {
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "display_name": item.get(
                "display_name",
                query,
            ),
        }

    except Exception:
        return None


# ============================================================
# NEARBY FACILITIES
# ============================================================

FACILITY_TAGS = {
    "👮 Police": [
        'node["amenity"="police"]',
        'way["amenity"="police"]',
        'relation["amenity"="police"]',
    ],
    "🏥 Hospital": [
        'node["amenity"="hospital"]',
        'way["amenity"="hospital"]',
        'relation["amenity"="hospital"]',
    ],
    "🏥 Clinic / Health Centre": [
        'node["amenity"="clinic"]',
        'way["amenity"="clinic"]',
        'node["healthcare"="centre"]',
        'way["healthcare"="centre"]',
    ],
    "💊 Pharmacy": [
        'node["amenity"="pharmacy"]',
        'way["amenity"="pharmacy"]',
    ],
    "🔥 Fire Station": [
        'node["amenity"="fire_station"]',
        'way["amenity"="fire_station"]',
    ],
    "⚖️ Legal Support": [
        'node["office"="lawyer"]',
        'way["office"="lawyer"]',
    ],
    "🟣 Women Support": [
        'node["social_facility"="shelter"]',
        'way["social_facility"="shelter"]',
        'node["social_facility"="outreach"]',
        'way["social_facility"="outreach"]',
    ],
    "👧 Child Support": [
        'node["amenity"="social_facility"]',
        'way["amenity"="social_facility"]',
    ],
}


def build_overpass_query(
    lat,
    lon,
    radius_m,
):

    parts = []

    for tags in FACILITY_TAGS.values():

        for tag in tags:
            parts.append(
                f'{tag}(around:{int(radius_m)},{lat},{lon});'
            )

    query = f"""
    [out:json][timeout:40];

    (
        {"".join(parts)}
    );

    out center tags;
    """

    return query


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def get_nearby_facilities(
    lat,
    lon,
    radius_km,
):

    radius_m = int(
        radius_km * 1000
    )

    query = build_overpass_query(
        lat,
        lon,
        radius_m,
    )

    headers = {
        "User-Agent": USER_AGENT
    }

    try:

        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers=headers,
            timeout=50,
        )

        response.raise_for_status()

        elements = response.json().get(
            "elements",
            [],
        )

    except Exception as exc:

        return [], str(exc)

    results = []

    for element in elements:

        tags = element.get(
            "tags",
            {},
        )

        latitude = element.get(
            "lat"
        )

        longitude = element.get(
            "lon"
        )

        if latitude is None:
            latitude = element.get(
                "center",
                {},
            ).get("lat")

        if longitude is None:
            longitude = element.get(
                "center",
                {},
            ).get("lon")

        if latitude is None or longitude is None:
            continue

        amenity = str(
            tags.get("amenity", "")
        ).lower()

        healthcare = str(
            tags.get("healthcare", "")
        ).lower()

        office = str(
            tags.get("office", "")
        ).lower()

        social = str(
            tags.get("social_facility", "")
        ).lower()

        if amenity == "police":
            category = "👮 Police"
        elif amenity == "hospital":
            category = "🏥 Hospital"
        elif (
            amenity == "clinic"
            or healthcare == "centre"
        ):
            category = "🏥 Clinic / Health Centre"
        elif amenity == "pharmacy":
            category = "💊 Pharmacy"
        elif amenity == "fire_station":
            category = "🔥 Fire Station"
        elif office == "lawyer":
            category = "⚖️ Legal Support"
        elif social in [
            "shelter",
            "outreach",
        ]:
            category = "🟣 Women Support"
        elif amenity == "social_facility":
            category = "👧 Child Support"
        else:
            category = "🏢 Support Facility"

        name = (
            tags.get("name")
            or tags.get("official_name")
            or "Unnamed facility"
        )

        address_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
            tags.get("addr:state"),
        ]

        address = ", ".join(
            str(x)
            for x in address_parts
            if x
        )

        distance = haversine_km(
            lat,
            lon,
            latitude,
            longitude,
        )

        results.append(
            {
                "name": name,
                "category": category,
                "lat": float(latitude),
                "lon": float(longitude),
                "distance_km": round(
                    distance,
                    2,
                ),
                "address": address or "Address not available",
                "phone": (
                    tags.get("phone")
                    or tags.get("contact:phone")