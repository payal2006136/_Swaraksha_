import math
import re
from difflib import get_close_matches

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None


# ============================================================
# SWARAKSHA
# Know the Pattern. Find the Help.
# ============================================================

st.set_page_config(
    page_title="Swaraksha | Know the Pattern. Find the Help.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #07111f;
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background: #0d1726;
}

.hero {
    padding: 20px 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0b1f36, #102b45);
    border: 1px solid #24445f;
    margin-bottom: 18px;
}

.hero h1 {
    margin: 0;
    color: #eaf6ff;
    font-size: 36px;
    font-weight: 800;
}

.hero p {
    margin-top: 7px;
    color: #b9cbe0;
    font-size: 15px;
}

.status-card {
    padding: 17px;
    border-radius: 15px;
    background: #101d2d;
    border: 1px solid #29415a;
    text-align: center;
    min-height: 100px;
}

.status-title {
    font-size: 13px;
    color: #9fb2c8;
}

.status-value {
    font-size: 24px;
    font-weight: 800;
    margin-top: 7px;
}

.info-card {
    padding: 15px;
    border-radius: 14px;
    background: #101d2d;
    border: 1px solid #29415a;
    margin-bottom: 12px;
}

.warning-card {
    padding: 13px;
    border-radius: 12px;
    background: #241d0d;
    border: 1px solid #705d25;
    color: #f5dfa1;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/raw/cleaned_community_safety_crime_2010_2022.csv"

USER_AGENT = "Swaraksha/1.0 civic-safety-streamlit"

NOMINATIM_URL = "https://nominatim.openstreetmap.org"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ============================================================
# CRIME CATEGORIES FROM ACTUAL DATASET
# ============================================================

CRIME_COLUMNS = {
    "Rape": "rape",
    "Assault on women": "assault_women",
    "Insult to modesty": "insult_modesty",
    "Importation of girls": "importation_girls",
    "Immoral traffic": "immoral_traffic",
    "Procuration of minor girls": "procuration_minor_girls",
    "Human trafficking": "human_trafficking",
    "Cyber explicit material": "cyber_explicit_material",
    "Other women cyber crimes": "other_women_cyber_crimes",
}


# ============================================================
# MAP FACILITY TYPES
# ============================================================

FACILITY_META = {

    "Police Station":
        ("👮", "red"),

    "Hospital":
        ("🏥", "green"),

    "Clinic / Health Centre":
        ("🩺", "lightgreen"),

    "Pharmacy":
        ("💊", "cadetblue"),

    "Fire Station":
        ("🔥", "orange"),

    "Women / Child Support":
        ("🟣", "purple"),

    "Shelter / Social Support":
        ("🏠", "darkpurple"),

    "Legal Support":
        ("⚖️", "darkblue"),
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    required = [
        "state",
        "district",
        "year",
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Required column '{col}' was not found."
            )

    df["state"] = (
        df["state"]
        .astype(str)
        .str.strip()
    )

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    for col in CRIME_COLUMNS.values():

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

            df[col] = df[col].clip(
                lower=0
            )

    df = df.dropna(
        subset=[
            "state",
            "district",
            "year"
        ]
    )

    df["year"] = df["year"].astype(int)

    return df


try:

    df = load_data()

except Exception as error:

    st.error(
        f"Dataset could not be loaded: {error}"
    )

    st.stop()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def norm_text(value):

    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

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

    earth_radius = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(dl / 2) ** 2
    )

    return (
        earth_radius
        *
        2
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(
                max(
                    0,
                    1 - a
                )
            )
        )
    )


# ============================================================
# LOW / MODERATE / HIGH CLASSIFICATION
# ============================================================

def classify(
    value,
    population
):

    population = pd.Series(
        population
    ).dropna()

    if population.empty:

        return (
            "🟡 MODERATE",
            "MODERATE"
        )

    q1 = population.quantile(0.33)
    q2 = population.quantile(0.67)

    if value <= q1:

        return (
            "🟢 LOW",
            "LOW"
        )

    if value >= q2:

        return (
            "🔴 HIGH",
            "HIGH"
        )

    return (
        "🟡 MODERATE",
        "MODERATE"
    )


# ============================================================
# GEOCODING
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def geocode_place(query):

    try:

        response = requests.get(

            f"{NOMINATIM_URL}/search",

            params={
                "q": f"{query}, India",
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "in",
                "addressdetails": 1,
            },

            headers={
                "User-Agent": USER_AGENT
            },

            timeout=12,
        )

        response.raise_for_status()

        data = response.json()

        if not data:

            return None

        result = data[0]

        return {

            "lat":
                float(result["lat"]),

            "lon":
                float(result["lon"]),

            "display":
                result.get(
                    "display_name",
                    query
                ),

            "address":
                result.get(
                    "address",
                    {}
                ),
        }

    except Exception:

        return None


# ============================================================
# REVERSE GEOCODING
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def reverse_geocode(
    lat,
    lon
):

    try:

        response = requests.get(

            f"{NOMINATIM_URL}/reverse",

            params={
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 18,
                "addressdetails": 1,
            },

            headers={
                "User-Agent": USER_AGENT
            },

            timeout=12,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get(
                "address",
                {}
            ),
            data.get(
                "display_name",
                ""
            )
        )

    except Exception:

        return {}, ""


# ============================================================
# DISTRICT FROM ADDRESS
# ============================================================

def district_from_address(
    address
):

    return (

        address.get(
            "state_district"
        )

        or address.get(
            "district"
        )

        or address.get(
            "county"
        )

        or address.get(
            "city_district"
        )

        or address.get(
            "city"
        )

        or address.get(
            "town"
        )

        or address.get(
            "municipality"
        )
    )


# ============================================================
# DISTRICT MATCHING
# ============================================================

def state_district_match(
    state_hint,
    district_hint
):

    if not district_hint:

        return None

    target = norm_text(
        district_hint
    )

    candidates = (
        df["district"]
        .dropna()
        .unique()
        .tolist()
    )

    exact = [

        x for x in candidates

        if norm_text(x) == target
    ]

    if exact:

        if state_hint:

            same_state = (

                df[
                    df["district"]
                    == exact[0]
                ]["state"]
                .unique()
                .tolist()
            )

            state_target = norm_text(
                state_hint
            )

            for state in same_state:

                normalized = norm_text(
                    state
                )

                if (
                    state_target == normalized
                    or state_target in normalized
                    or normalized in state_target
                ):

                    return exact[0]

        return exact[0]

    normalized_candidates = [

        norm_text(x)
        for x in candidates
    ]

    close = get_close_matches(
        target,
        normalized_candidates,
        n=1,
        cutoff=0.75
    )

    if close:

        return next(

            (
                x
                for x in candidates
                if norm_text(x)
                == close[0]
            ),

            None
        )

    return None


# ============================================================
# LIVE SUPPORT FACILITIES
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_nearby_facilities(
    lat,
    lon,
    radius_m
):

    query = f"""
    [out:json][timeout:35];

    (
        nwr["amenity"="police"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="hospital"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="clinic"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="pharmacy"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="fire_station"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="social_facility"]
            (around:{radius_m},{lat},{lon});

        nwr["amenity"="shelter"]
            (around:{radius_m},{lat},{lon});

        nwr["office"="lawyer"]
            (around:{radius_m},{lat},{lon});

        nwr["social_facility"="shelter"]
            (around:{radius_m},{lat},{lon});

        nwr["name"~"One Stop|One-Stop|Sakhi|Shakti Sadan|Child Care|Child Support|Women.*Centre|Women.*Center|Women Support|Child Protection|Legal Aid|NALSA",i]
            (around:{radius_m},{lat},{lon});
    );

    out center tags;
    """

    try:

        response = requests.post(

            OVERPASS,

            data={
                "data": query
            },

            headers={
                "User-Agent":
                    USER_AGENT
            },

            timeout=45,
        )

        response.raise_for_status()

        elements = (
            response.json()
            .get(
                "elements",
                []
            )
        )

    except Exception:

        return []

    results = []

    seen = set()

    for element in elements:

        tags = element.get(
            "tags",
            {}
        )

        element_lat = (

            element.get("lat")

            or element
            .get("center", {})
            .get("lat")
        )

        element_lon = (

            element.get("lon")

            or element
            .get("center", {})
            .get("lon")
        )

        if (
            element_lat is None
            or element_lon is None
        ):

            continue

        element_lat = float(
            element_lat
        )

        element_lon = float(
            element_lon
        )

        name = (

            tags.get("name")

            or tags.get(
                "official_name"
            )
        )

        if not name:

            continue

        amenity = (
            tags.get(
                "amenity",
                ""
            )
            .lower()
        )

        office = (
            tags.get(
                "office",
                ""
            )
            .lower()
        )

        social = (
            tags.get(
                "social_facility",
                ""
            )
            .lower()
        )

        text = (
            f"{name} "
            f"{tags.get('description', '')} "
            f"{tags.get('operator', '')} "
            f"{tags.get('service', '')}"
        ).lower()

        # ----------------------------------------------------
        # CLASSIFY FACILITY
        # ----------------------------------------------------

        if amenity == "police":

            category = "Police Station"

        elif amenity == "hospital":

            category = "Hospital"

        elif amenity == "clinic":

            category = (
                "Clinic / Health Centre"
            )

        elif amenity == "pharmacy":

            category = "Pharmacy"

        elif amenity == "fire_station":

            category = "Fire Station"

        elif (

            office == "lawyer"

            or any(

                keyword in text

                for keyword in [

                    "legal aid",
                    "legal service",
                    "lawyer",
                    "nalsa",
                    "legal support",
                ]
            )
        ):

            category = "Legal Support"

        elif any(

            keyword in text

            for keyword in [

                "one stop",
                "one-stop",
                "sakhi",
                "shakti sadan",
                "women support",
                "women centre",
                "women center",
                "child support",
                "child care",
                "child protection",
            ]
        ):

            category = (
                "Women / Child Support"
            )

        elif (

            amenity in {
                "social_facility",
                "shelter",
            }

            or social
        ):

            category = (
                "Shelter / Social Support"
            )

        else:

            continue

        distance = haversine_km(

            lat,
            lon,
            element_lat,
            element_lon
        )

        if distance * 1000 > radius_m:

            continue

        address_parts = [

            tags.get(
                "addr:housenumber",
                ""
            ),

            tags.get(
                "addr:street",
                ""
            ),

            tags.get(
                "addr:suburb",
                ""
            ),

            (
                tags.get(
                    "addr:city",
                    ""
                )

                or tags.get(
                    "addr:town",
                    ""
                )
            ),
        ]

        address = ", ".join(

            part

            for part in address_parts

            if part
        )

        if not address:

            address = (
                "Address not mapped"
            )

        key = (

            norm_text(name),

            category,

            round(
                element_lat,
                4
            ),

            round(
                element_lon,
                4
            ),
        )

        if key in seen:

            continue

        seen.add(key)

        results.append({

            "name": name,

            "category": category,

            "lat": element_lat,

            "lon": element_lon,

            "distance_km":
                round(
                    distance,
                    2
                ),

            "address": address,

            "phone":
                tags.get("phone")
                or tags.get(
                    "contact:phone"
                )
                or "Not mapped",

            "website":
                tags.get("website")
                or tags.get(
                    "contact:website"
                )
                or "",
        })

    return sorted(
        results,
        key=lambda item:
            item["distance_km"]
    )


# ============================================================
# CRIME ANALYSIS
# ============================================================

def district_frame(
    district
):

    return df[
        df["district"]
        .apply(norm_text)
        == norm_text(district)
    ].copy()


def category_status(
    district_df,
    column
):

    if (
        column not in district_df.columns
        or district_df.empty
    ):

        return None

    district_annual = (
        district_df
        .groupby("year")[column]
        .sum()
    )

    district_mean = (
        district_annual.mean()
    )

    national_means = (

        df.groupby(
            ["district", "year"]
        )[column]
        .sum()
        .groupby("district")
        .mean()
    )

    label, short = classify(

        district_mean,

        national_means.values
    )

    return {

        "mean":
            float(district_mean),

        "label":
            label,

        "short":
            short,
    }


# ============================================================
# OVERALL HISTORICAL INDICATOR
# ============================================================

def overall_historical(
    district_df
):

    statuses = []

    for label, column in CRIME_COLUMNS.items():

        if column not in district_df.columns:

            continue

        result = category_status(
            district_df,
            column
        )

        if result:

            statuses.append(
                result
            )

    if not statuses:

        return (
            "🟡 MODERATE",
            "MODERATE"
        )

    score = np.mean([

        0
        if x["short"] == "LOW"

        else 1
        if x["short"] == "MODERATE"

        else 2

        for x in statuses

    ])

    if score < 0.67: