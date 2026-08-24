"""
🛡️ SWARAKSHA — Know the Pattern. Find the Help.
A location-first community safety web app.

Core principle:
- Historical crime data is used only for a district-level historical pattern indicator.
- Current nearby facilities are discovered from live public geographic services.
- No street-level crime hotspots are invented.
- No area is guaranteed safe or dangerous.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

# Optional browser geolocation component.
try:
    from streamlit_geolocation import streamlit_geolocation
    GEOLOCATION_AVAILABLE = True
except Exception:
    GEOLOCATION_AVAILABLE = False


# ---------------------------------------------------------------------
# APP CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Swaraksha | Know the Pattern. Find the Help.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "cleaned_community_safety_crime_2010_2022.csv"
DB_PATH = BASE_DIR / "database" / "swaraksha_feedback.db"

# The dataset actually contains 2010–2022. We use 2017–2022 as the
# default project-analysis window because that is the requested period,
# while preserving the complete file coverage.
DEFAULT_START_YEAR = 2017
DEFAULT_END_YEAR = 2022

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

USER_AGENT = "SwarakshaSafetyProject/1.0 (academic civic-tech application)"

# Only the helpline explicitly verified for this environment is hard-coded.
# Additional numbers can be added after verification from official sources.
HELPLINES = [
    {
        "group": "Mental / Emotional Support",
        "icon": "🧠",
        "name": "Tele-MANAS",
        "number": "14416",
        "note": "Government of India mental-health support.",
    }
]

# Dataset columns actually present in the uploaded CSV.
WOMEN_COLUMNS = [
    "rape",
    "assault_women",
    "insult_modesty",
    "importation_girls",
    "immoral_traffic",
    "procuration_minor_girls",
    "human_trafficking",
    "cyber_explicit_material",
    "other_women_cyber_crimes",
]

CHILD_COLUMNS = [
    "assault_women_below18",
    "insult_modesty_below18",
    "procuration_minor_girls",
]

# ---------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --navy:#10233f;
    --blue:#1769aa;
    --teal:#0f766e;
    --light:#f5f8fc;
    --border:#dfe7f1;
    --text:#172033;
}
.block-container { padding-top: 1.2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #f7f9fc; }
.hero {
    padding: 28px 30px;
    border-radius: 24px;
    background: linear-gradient(135deg,#10233f 0%,#1769aa 55%,#0f766e 100%);
    color: white;
    margin-bottom: 22px;
}
.hero h1 { margin: 0 0 8px 0; font-size: 42px; }
.hero p { margin: 4px 0; font-size: 17px; opacity: .94; }
.card {
    background:white;
    border:1px solid #dfe7f1;
    border-radius:18px;
    padding:18px;
    margin-bottom:14px;
    box-shadow:0 5px 20px rgba(16,35,63,.05);
}
.feature {
    background:#f7faff;
    border:1px solid #e4edf7;
    border-radius:18px;
    padding:18px;
    min-height:120px;
}
.metric {
    background:#fff;
    border:1px solid #dfe7f1;
    border-radius:16px;
    padding:16px;
    text-align:center;
}
.metric .num { font-size:30px; font-weight:800; color:#10233f; }
.metric .lbl { color:#58657a; font-size:14px; }
.disclaimer {
    background:#fff7e6;
    border-left:5px solid #f0a202;
    border-radius:10px;
    padding:12px 15px;
    margin:10px 0 18px 0;
}
.low { color:#137333; font-weight:800; }
.moderate { color:#9a6700; font-weight:800; }
.high { color:#b3261e; font-weight:800; }
.support-row {
    background:#ffffff;
    border:1px solid #e2e8f0;
    border-radius:14px;
    padding:12px;
    margin:7px 0;
}
.small { color:#667085; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_num(value: Any) -> float:
    try:
        x = float(value)
        return 0.0 if np.isnan(x) else x
    except Exception:
        return 0.0


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def format_distance(km: float) -> str:
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.2f} km"


def indicator_class(label: str) -> str:
    return {"LOW": "low", "MODERATE": "moderate", "HIGH": "high"}.get(label, "")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                category TEXT NOT NULL,
                description TEXT,
                rating INTEGER
            )
            """
        )
        con.commit()


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected data/raw/cleaned_community_safety_crime_2010_2022.csv"
        )

    df = pd.read_csv(path)

    # Standardize only fields that are actually expected in this dataset.
    for col in ["state", "district"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().str.upper()

    if "year" not in df.columns:
        raise ValueError("The dataset must contain a 'year' column.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    numeric_cols = [
        c for c in WOMEN_COLUMNS + CHILD_COLUMNS + ["total_crimes", "total_community_crimes"]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    return df


def relevant_women_total(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in WOMEN_COLUMNS if c in frame.columns]
    return frame[cols].sum(axis=1) if cols else pd.Series(0, index=frame.index)


def relevant_child_total(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in CHILD_COLUMNS if c in frame.columns]
    return frame[cols].sum(axis=1) if cols else pd.Series(0, index=frame.index)


def district_indicator(
    df: pd.DataFrame,
    state: str,
    district: str,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> Dict[str, Any]:
    period = df[df["year"].between(start_year, end_year)].copy()
    target = period[(period["state"] == norm_text(state)) & (period["district"] == norm_text(district))]

    if target.empty:
        return {
            "label": "LOW",
            "score": 0.0,
            "target_total": 0.0,
            "peer_percentile": None,
            "factors": [],
            "available": False,
        }

    target_women = float(relevant_women_total(target).sum())
    target_child = float(relevant_child_total(target).sum())
    target_total = target_women + target_child

    # Peer comparison is district-level and based only on reported counts.
    # No population denominator is available in this dataset, so this is NOT
    # a per-capita risk score.
    grouped = period.groupby(["state", "district"], dropna=False)
    peer = grouped.apply(
        lambda x: float(relevant_women_total(x).sum() + relevant_child_total(x).sum()),
        include_groups=False,
    )
    peer = pd.to_numeric(peer, errors="coerce").fillna(0)

    percentile = float((peer <= target_total).mean() * 100) if len(peer) else 0.0

    if percentile >= 75:
        label = "HIGH"
    elif percentile >= 45:
        label = "MODERATE"
    else:
        label = "LOW"

    factors = []
    factor_values = {}
    for col in WOMEN_COLUMNS:
        if col in target.columns:
            value = float(target[col].sum())
            factor_values[col] = value

    for col, value in sorted(factor_values.items(), key=lambda x: x[1], reverse=True)[:4]:
        if value > 0:
            factors.append((col.replace("_", " ").title(), value))

    return {
        "label": label,
        "score": percentile,
        "target_total": target_total,
        "women_total": target_women,
        "child_total": target_child,
        "peer_percentile": percentile,
        "factors": factors,
        "available": True,
    }


# ---------------------------------------------------------------------
# GEOCODING
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(query: str) -> Optional[Dict[str, Any]]:
    if not query.strip():
        return None
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": f"{query}, India", "format": "jsonv2", "limit": 1, "countrycodes": "in"},
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        item = results[0]
        return {
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "display_name": item.get("display_name", query),
            "type": item.get("type", ""),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "addressdetails": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        addr = data.get("address", {})
        return {
            "display_name": data.get("display_name", "Current location"),
            "state": addr.get("state", ""),
            "district": (
                addr.get("state_district")
                or addr.get("district")
                or addr.get("county")
                or ""
            ),
            "city": addr.get("city") or addr.get("town") or addr.get("municipality") or addr.get("village") or "",
        }
    except Exception:
        return None


def match_state_district(
    df: pd.DataFrame,
    state_text: str,
    district_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    state_text = norm_text(state_text)
    district_text = norm_text(district_text)

    if not state_text or not district_text:
        return None, None

    states = df["state"].dropna().unique().tolist()
    exact_state = next((s for s in states if s == state_text), None)
    if exact_state is None:
        # Conservative substring matching.
        candidates = [s for s in states if state_text in s or s in state_text]
        exact_state = candidates[0] if len(candidates) == 1 else None

    if exact_state is None:
        return None, None

    districts = df.loc[df["state"] == exact_state, "district"].dropna().unique().tolist()
    exact_district = next((d for d in districts if d == district_text), None)
    if exact_district is None:
        candidates = [d for d in districts if district_text in d or d in district_text]
        exact_district = candidates[0] if len(candidates) == 1 else None

    return exact_state, exact_district


# ---------------------------------------------------------------------
# LIVE NEARBY FACILITIES
# ---------------------------------------------------------------------
FACILITY_QUERIES = {
    "all": [
        'nwr["amenity"="police"]',
        'nwr["amenity"="hospital"]',
        'nwr["amenity"="clinic"]',
        'nwr["amenity"="pharmacy"]',
        'nwr["amenity"="fire_station"]',
        'nwr["amenity"="ambulance_station"]',
        'nwr["office"="lawyer"]',
        'nwr["social_facility"~"shelter|childcare|outreach"]',
        'nwr["amenity"="social_centre"]',
        'nwr["amenity"="community_centre"]',
        'nwr["name"~"One Stop Centre|One Stop Center|Sakhi|Shakti Sadan",i]',
    ],
    "police": ['nwr["amenity"="police"]'],
    "medical": [
        'nwr["amenity"="hospital"]',
        'nwr["amenity"="clinic"]',
        'nwr["amenity"="pharmacy"]',
    ],
    "support": [
        'nwr["office"="lawyer"]',
        'nwr["social_facility"~"shelter|childcare|outreach"]',
        'nwr["amenity"="social_centre"]',
        'nwr["name"~"One Stop Centre|One Stop Center|Sakhi|Shakti Sadan",i]',
    ],
}


def classify_osm(tags: Dict[str, Any]) -> Tuple[str, str]:
    name = str(tags.get("name", "Unnamed facility"))
    amenity = str(tags.get("amenity", ""))
    office = str(tags.get("office", ""))
    social = str(tags.get("social_facility", ""))

    if amenity == "police":
        return "police", "👮 Police"
    if amenity == "hospital":
        return "hospital", "🏥 Hospital"
    if amenity in {"clinic", "doctors"}:
        return "clinic", "🏥 Clinic / Health Centre"
    if amenity == "pharmacy":
        return "pharmacy", "💊 Pharmacy"
    if amenity == "fire_station":
        return "fire", "🔥 Fire Station"
    if amenity == "ambulance_station":
        return "ambulance", "🚑 Ambulance"
    if office == "lawyer":
        return "legal", "⚖️ Legal / Lawyer"
    if social in {"shelter", "childcare", "outreach"}:
        return "support", "🏠 Social / Child Support"
    if amenity in {"social_centre", "community_centre"}:
        return "support", "🏠 Community Support"
    if re.search(r"one stop centre|one stop center|sakhi|shakti sadan", name, re.I):
        return "women_support", "🟣 Women Support"
    return "support", "🏠 Support"


@st.cache_data(ttl=600, show_spinner=False)
def get_nearby_facilities(lat: float, lon: float, radius_m: int = 5000, category: str = "all") -> List[Dict[str, Any]]:
    tags = FACILITY_QUERIES.get(category, FACILITY_QUERIES["all"])
    query_parts = "\n".join(f"  {tag}(around:{int(radius_m)},{lat},{lon});" for tag in tags)
    query = f"""
    [out:json][timeout:25];
    (
    {query_parts}
    );
    out center tags;
    """

    last_error = None
    for endpoint in OVERPASS_URLS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=35,
            )
            r.raise_for_status()
            elements = r.json().get("elements", [])
            results: List[Dict[str, Any]] = []
            seen = set()

            for el in elements:
                e_lat = el.get("lat") or el.get("center", {}).get("lat")
                e_lon = el.get("lon") or el.get("center", {}).get("lon")
                if e_lat is None or e_lon is None:
                    continue

                tags_dict = el.get("tags", {})
                name = tags_dict.get("name") or tags_dict.get("official_name") or "Unnamed facility"
                kind, label = classify_osm(tags_dict)

                # Avoid repeated node/way duplicates with same name and rounded location.
                key = (name.lower(), round(float(e_lat), 5), round(float(e_lon), 5))
                if key in seen:
                    continue
                seen.add(key)

                distance = haversine_km(lat, lon, float(e_lat), float(e_lon))
                address_parts = [
                    tags_dict.get("addr:housenumber", ""),
                    tags_dict.get("addr:street", ""),
                    tags_dict.get("addr:suburb", ""),
                    tags_dict.get("addr:city", ""),
                ]
                address = ", ".join(p for p in address_parts if p)
                phone = tags_dict.get("phone") or tags_dict.get("contact:phone")
                website = tags_dict.get("website") or tags_dict.get("contact:website")

                results.append(
                    {
                        "name": name,
                        "type": kind,
                        "label": label,
                        "latitude": float(e_lat),
                        "longitude": float(e_lon),
                        "distance_km": round(distance, 3),
                        "address": address or "Address not mapped",
                        "phone": phone or "",
                        "website": website or "",
                    }
                )

            return sorted(results, key=lambda x: x["distance_km"])
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Live facility search is temporarily unavailable: {last_error}")


# ---------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------
def build_map(
    center_lat: float = 22.5937,
    center_lon: float = 78.9629,
    zoom: int = 5,
    user_location: Optional[Tuple[float, float]] = None,
    facilities: Optional[List[Dict[str, Any]]] = None,
) -> folium.Map:
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    Fullscreen(position="topright").add_to(m)

    if user_location:
        ulat, ulon = user_location
        folium.Marker(
            [ulat, ulon],
            tooltip="📍 Your selected/current location",
            popup=f"Latitude: {ulat:.6f}<br>Longitude: {ulon:.6f}",
            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
        ).add_to(m)

        folium.Circle(
            [ulat, ulon],
            radius=5000,
            color="#1769aa",
            fill=True,
            fill_opacity=0.05,
            tooltip="5 km nearby-support search radius",
        ).add_to(m)

    for item in facilities or []:
        icon_color = {
            "police": "red",
            "hospital": "green",
            "clinic": "green",
            "pharmacy": "purple",
            "fire": "orange",
            "ambulance": "darkgreen",
            "legal": "blue",
            "support": "cadetblue",
            "women_support": "pink",
        }.get(item["type"], "gray")

        popup = (
            f"<b>{item['label']} — {item['name']}</b><br>"
            f"Distance: {format_distance(item['distance_km'])}<br>"
            f"{item['address']}"
        )
        if item.get("phone"):
            popup += f"<br>Phone: {item['phone']}"

        folium.Marker(
            [item["latitude"], item["longitude"]],
            tooltip=f"{item['label']}: {item['name']}",
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(m)

    return m


# ---------------------------------------------------------------------
# UI STATE / LOCATION
# ---------------------------------------------------------------------
def save_location(lat: float, lon: float, source: str) -> None:
    st.session_state["lat"] = float(lat)
    st.session_state["lon"] = float(lon)
    st.session_state["location_source"] = source
    st.session_state["reverse"] = reverse_geocode(float(lat), float(lon))


def current_location_widget() -> None:
    if not GEOLOCATION_AVAILABLE:
        st.warning(
            "Browser GPS component is not installed. Use the manual location search below."
        )
        return

    try:
        location = streamlit_geolocation()
        if location and location.get("latitude") is not None and location.get("longitude") is not None:
            lat = float(location["latitude"])
            lon = float(location["longitude"])
            if (
                "lat" not in st.session_state
                or abs(st.session_state["lat"] - lat) > 0.00001
                or abs(st.session_state["lon"] - lon) > 0.00001
            ):
                save_location(lat, lon, "browser GPS")
            st.success(
                f"📍 Live location received: {lat:.5f}, {lon:.5f} "
                f"(accuracy shown by your browser/device)."
            )
    except Exception as exc:
        st.info(f"GPS permission/availability issue: {exc}")


def location_search_box() -> None:
    st.markdown("### 🔎 Search a place instead")
    query = st.text_input(
        "Indian address / locality / landmark",
        placeholder="Example: Thane West, Maharashtra",
        key="location_query",
    )
    if st.button("Find this location", use_container_width=True):
        if not query.strip():
            st.warning("Enter a location first.")
            return
        with st.spinner("Finding location…"):
            result = geocode_place(query)
        if result:
            save_location(result["lat"], result["lon"], "manual search")
            st.success(f"📍 Found: {result['display_name']}")
            st.rerun()
        else:
            st.error("Location not found. Try a more specific Indian address.")


def location_context(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    rev = st.session_state.get("reverse") or {}
    state_text = rev.get("state", "")
    district_text = rev.get("district", "")
    return match_state_district(df, state_text, district_text)


# ---------------------------------------------------------------------
# SMALL UI COMPONENTS
# ---------------------------------------------------------------------
def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🛡️ SWARAKSHA</h1>
          <p><b>Know the Pattern. Find the Help.</b></p>
          <p>India-focused location assistance combining historical crime patterns with current nearby support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer"><b>Important:</b> Historical crime patterns do not guarantee present or future safety. Swaraksha does not declare any place definitely safe or dangerous.</div>',
        unsafe_allow_html=True,
    )


def indicator_card(result: Dict[str, Any]) -> None:
    label = result["label"]
    emoji = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}[label]
    st.markdown(
        f"""
        <div class="card">
          <div class="{indicator_class(label)}" style="font-size:28px">{emoji} {label} HISTORICAL PATTERN</div>
          <div class="small">Relative reported-crime pattern for the selected district and analysis period. This is not a live danger score.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(items: List[Tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="metric"><div class="num">{value}</div><div class="lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )


def facility_cards(facilities: List[Dict[str, Any]], limit: int = 12) -> None:
    if not facilities:
        st.info(
            "No mapped facilities were returned for this search radius. "
            "This does not prove that no facility exists."
        )
        return

    for item in facilities[:limit]:
        phone = f" · ☎️ {item['phone']}" if item.get("phone") else ""
        st.markdown(
            f"""
            <div class="support-row">
              <b>{item['label']} — {item['name']}</b><br>
              <span class="small">{format_distance(item['distance_km'])} · {item['address']}{phone}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------
def home_page(df: pd.DataFrame) -> None:
    hero()

    st.markdown("## 📍 Start with your location")
    st.write(
        "Swaraksha is designed to answer two practical questions: "
        "**what does historical data say about this district?** and "
        "**what current support is nearby?**"
    )

    current_location_widget()

    if "lat" in st.session_state:
        lat, lon = st.session_state["lat"], st.session_state["lon"]
        rev = st.session_state.get("reverse") or {}
        st.markdown("### 📌 Your selected/current location")
        metric_cards(
            [
                ("Latitude", f"{lat:.5f}"),
                ("Longitude", f"{lon:.5f}"),
                ("Source", st.session_state.get("location_source", "location")),
            ]
        )
        if rev.get("display_name"):
            st.caption(rev["display_name"])

        state, district = location_context(df)
        if state and district:
            result = district_indicator(df, state, district)
            st.markdown(f"### {state.title()} · {district.title()}")
            indicator_card(result)
            st.caption(
                "The district match comes from reverse geocoding and the uploaded dataset. "
                "A street-level crime hotspot is not inferred."
            )
        else:
            st.info(
                "Your coordinates were received, but the location could not be confidently matched "
                "to a district in the uploaded dataset. Nearby-support search can still work."
            )

        if st.button("🆘 Find Help Near Me", type="primary", use_container_width=True):
            st.session_state["page"] = "🆘 Help Nearby"
            st.rerun()

    location_search_box()

    st.markdown("## 🧭 What you can do")
    features = [
        ("📍", "Check an Area", "Search a place or use your current location."),
        ("🗺️", "Live GIS Map", "See your location and current mapped support nearby."),
        ("👮", "Police", "Find mapped police stations and distance."),
        ("🏥", "Medical Help", "Find mapped hospitals, clinics and pharmacies."),
        ("🟣", "Women Support", "Look for mapped women-support facilities where available."),
        ("🆘", "Help Nearby", "Search multiple support categories around you."),
    ]
    cols = st.columns(3)
    for i, (icon, title, text) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f'<div class="feature"><h3>{icon} {title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True,
            )


def check_area_page(df: pd.DataFrame) -> None:
    hero()
    st.markdown("## 📍 Check an Area")

    states = sorted(df["state"].dropna().unique().tolist())
    selected_state = st.selectbox("State / UT", states)
    districts = sorted(df.loc[df["state"] == selected_state, "district"].dropna().unique().tolist())
    selected_district = st.selectbox("District", districts)

    result = district_indicator(df, selected_state, selected_district)
    indicator_card(result)

    period = df[df["year"].between(DEFAULT_START_YEAR, DEFAULT_END_YEAR)]
    target = period[
        (period["state"] == selected_state)
        & (period["district"] == selected_district)
    ]

    if target.empty:
        st.warning("No records found for this selection.")
        return

    metric_cards(
        [
            ("Data records", len(target)),
            ("Years in file", f"{int(target['year'].min())}–{int(target['year'].max())}"),
            ("Women-related reports", int(relevant_women_total(target).sum())),
            ("Available under-18 fields", int(relevant_child_total(target).sum())),
        ]
    )

    st.markdown("### 📈 Historical pattern")
    st.caption("Default analysis window: 2017–2022. The uploaded file itself contains 2010–2022.")

    yearly = target.groupby("year", as_index=False).agg(
        rape=("rape", "sum"),
        assault_women=("assault_women", "sum"),
        insult_modesty=("insult_modesty", "sum"),
        cyber_explicit_material=("cyber_explicit_material", "sum"),
        other_women_cyber_crimes=("other_women_cyber_crimes", "sum"),
    )
    long = yearly.melt("year", var_name="crime_type", value_name="reported_cases")
    long["crime_type"] = long["crime_type"].str.replace("_", " ").str.title()
    fig = px.line(
        long,
        x="year",
        y="reported_cases",
        color="crime_type",
        markers=True,
        title="Yearly reported crime indicators",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        women_values = {
            col.replace("_", " ").title(): int(target[col].sum())
            for col in WOMEN_COLUMNS
            if col in target.columns and target[col].sum() > 0
        }
        if women_values:
            fig2 = px.bar(
                x=list(women_values.keys()),
                y=list(women_values.values()),
                title="Women-related reported indicators",
                labels={"x": "Category", "y": "Reported cases"},
            )
            fig2.update_xaxes(tickangle=-35)
            st.plotly_chart(fig2, use_container_width=True)
    with c2:
        child_values = {
            col.replace("_", " ").title(): int(target[col].sum())
            for col in CHILD_COLUMNS
            if col in target.columns and target[col].sum() > 0
        }
        if child_values:
            fig3 = px.bar(
                x=list(child_values.keys()),
                y=list(child_values.values()),
                title="Available under-18 related indicators",
                labels={"x": "Category", "y": "Reported cases"},
            )
            fig3.update_xaxes(tickangle=-35)
            st.plotly_chart(fig3, use_container_width=True)

    if result["factors"]:
        st.markdown("### 🔎 What contributes to this historical pattern?")
        for name, value in result["factors"]:
            st.write(f"• **{name}:** {int(value):,} reported cases in the selected period.")

    st.markdown("### 🗺️ Find current help around this district")
    if st.button("📍 Find nearby help for this district", use_container_width=True):
        # District centroid is not available in the CSV. We geocode the district
        # instead of inventing coordinates.
        with st.spinner("Finding the district on the live map…"):
            result_geo = geocode_place(f"{selected_district}, {selected_state}")
        if result_geo:
            save_location(result_geo["lat"], result_geo["lon"], "district search")
            st.session_state["page"] = "🆘 Help Nearby"
            st.rerun()
        else:
            st.error("The district could not be geocoded. Try a nearby locality/landmark.")


def help_nearby_page(df: pd.DataFrame) -> None:
    hero()
    st.markdown("## 🆘 Find Help Near Me")

    if "lat" not in st.session_state:
        st.info("Start by allowing location access or searching an address.")
        current_location_widget()
        location_search_box()
        return

    lat, lon = st.session_state["lat"], st.session_state["lon"]
    rev = st.session_state.get("reverse") or {}

    st.success(
        f"📍 Using {st.session_state.get('location_source', 'selected location')}: "
        f"{lat:.5f}, {lon:.5f}"
    )
    if rev.get("display_name"):
        st.caption(rev["display_name"])

    state, district = location_context(df)
    if state and district:
        result = district_indicator(df, state, district)
        indicator_card(result)
        st.caption(f"Matched district: {state.title()} · {district.title()}")
    else:
        st.info("Historical district indicator is unavailable for this exact location match.")

    radius_km = st.slider("Search radius", 1, 10, 5)
    category = st.selectbox(
        "What help do you need?",
        [
            ("all", "🆘 All mapped support"),
            ("police", "👮 Police"),
            ("medical", "🏥 Medical"),
            ("support", "🏠 Women / social / legal support"),
        ],
        format_func=lambda x: x[1],
    )[0]

    if st.button("🔎 Search live nearby facilities", type="primary", use_container_width=True):
        with st.spinner("Searching current mapped facilities…"):
            try:
                facilities = get_nearby_facilities(lat, lon, radius_km * 1000, category)
                st.session_state["facilities"] = facilities
            except Exception as exc:
                st.error(str(exc))
                st.session_state["facilities"] = []

    facilities = st.session_state.get("facilities", [])

    metric_counts = []
    for typ, label in [
        ("police", "👮 Police"),
        ("hospital", "🏥 Hospitals"),
        ("clinic", "🏥 Clinics"),
        ("pharmacy", "💊 Pharmacies"),
        ("legal", "⚖️ Legal"),
        ("women_support", "🟣 Women support"),
    ]:
        metric_counts.append((label, sum(1 for f in facilities if f["type"] == typ)))
    metric_cards(metric_counts)

    st.markdown("### 🗺️ Live GIS map")
    m = build_map(lat, lon, 14, (lat, lon), facilities)
    st_folium(m, height=600, use_container_width=True)

    st.markdown("### 📋 Nearest mapped facilities")
    facility_cards(facilities)


def women_page(df: pd.DataFrame) -> None:
    hero()
    st.markdown("## 👩 Women Safety")

    states = sorted(df["state"].unique().tolist())
    state = st.selectbox("State / UT", states, key="women_state")
    districts = sorted(df.loc[df["state"] == state, "district"].unique().tolist())
    district = st.selectbox("District", districts, key="women_district")
    target = df[
        df["year"].between(DEFAULT_START_YEAR, DEFAULT_END_YEAR)
        & (df["state"] == state)
        & (df["district"] == district)
    ]
    if target.empty:
        st.warning("No data available.")
        return

    result = district_indicator(df, state, district)
    indicator_card(result)

    yearly = target.groupby("year", as_index=False).apply(
        lambda x: pd.Series({"women_related": relevant_women_total(x).sum()}),
        include_groups=False,
    ).reset_index(drop=True)
    fig = px.line(yearly, x="year", y="women_related", markers=True, title="Women-related reported indicators")
    st.plotly_chart(fig, use_container_width=True)

    vals = {
        c.replace("_", " ").title(): int(target[c].sum())
        for c in WOMEN_COLUMNS if c in target.columns and target[c].sum() > 0
    }
    if vals:
        fig2 = px.bar(x=list(vals.keys()), y=list(vals.values()), title="Available women-related categories")
        fig2.update_xaxes(tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "This page describes historical reported patterns in the uploaded data. "
        "It does not predict whether an individual person will be safe."
    )


def child_page(df: pd.DataFrame) -> None:
    hero()
    st.markdown("## 👧 Child Safety")

    states = sorted(df["state"].unique().tolist())
    state = st.selectbox("State / UT", states, key="child_state")
    districts = sorted(df.loc[df["state"] == state, "district"].unique().tolist())
    district = st.selectbox("District", districts, key="child_district")
    target = df[
        df["year"].between(DEFAULT_START_YEAR, DEFAULT_END_YEAR)
        & (df["state"] == state)
        & (df["district"] == district)
    ]

    vals = {
        c.replace("_", " ").title(): int(target[c].sum())
        for c in CHILD_COLUMNS if c in target.columns and target[c].sum() > 0
    }
    if vals:
        fig = px.bar(
            x=list(vals.keys()),
            y=list(vals.values()),
            title="Available under-18 / child-related indicators",
        )
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usable under-18-specific fields are populated for this selection.")

    st.warning(
        "The uploaded dataset does not contain a dedicated POCSO column. "
        "Swaraksha will not pretend that another column is POCSO."
    )


def map_page(df: pd.DataFrame) -> None:
    hero()
    st.markdown("## 🗺️ Live GIS Safety Map")

    lat = st.session_state.get("lat")
    lon = st.session_state.get("lon")
    facilities = st.session_state.get("facilities", [])

    if lat is None or lon is None:
        st.info("The map opens over India by default. Allow GPS or search a place to focus on your location.")
        m = build_map()
        st_folium(m, height=650, use_container_width=True)
    else:
        m = build_map(lat, lon, 14, (lat, lon), facilities)
        st_folium(m, height=650, use_container_width=True)

    st.caption(
        "Basemap and mapped facilities come from OpenStreetMap-compatible geographic services. "
        "Mapped coverage can be incomplete or outdated."
    )


def helpline_page() -> None:
    hero()
    st.markdown("## 🚨 Emergency & Helplines")
    st.info("For immediate danger, use your local emergency service or contact a trusted person. Swaraksha does not dispatch emergency services.")

    for group in ["Emergency", "Women", "Children", "Cyber Crime", "Legal Aid", "Mental / Emotional Support"]:
        entries = [x for x in HELPLINES if x["group"] == group]
        if not entries:
            continue
        st.markdown(f"### {group}")
        for item in entries:
            st.markdown(
                f'<div class="card"><h3>{item["icon"]} {item["name"]}</h3>'
                f'<div style="font-size:28px;font-weight:800">📞 {item["number"]}</div>'
                f'<div class="small">{item["note"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### ⚠️ Verification note")
    st.caption(
        "Only a helpline explicitly verified for this deployment is shown in the code. "
        "Additional government numbers should be added only after checking the current official source."
    )


def feedback_page() -> None:
    hero()
    st.markdown("## 👥 Community Feedback")
    st.caption(
        "Feedback is stored without names, phone numbers or other unnecessary personal information. "
        "One report does not immediately change the historical indicator."
    )

    categories = [
        "💡 Poor lighting",
        "🏚️ Isolated area",
        "👥 Low public activity",
        "🚌 Transport concern",
        "⚠️ Harassment concern",
        "🏥 Poor access to support",
        "👮 Visible police presence",
        "💡 Good lighting",
        "👥 Active/public area",
        "➕ Other",
    ]
    category = st.selectbox("What would you like to report?", categories)
    description = st.text_area("Optional description", max_chars=500)
    rating = st.slider("How concerned did you feel?", 1, 5, 3)

    lat = st.session_state.get("lat")
    lon = st.session_state.get("lon")
    if lat is not None:
        st.caption(f"Approximate selected location: {lat:.5f}, {lon:.5f}")
    else:
        st.caption("No location will be stored unless you have selected a location in this session.")

    if st.button("Submit feedback", type="primary", use_container_width=True):
        try:
            with sqlite3.connect(DB_PATH) as con:
                con.execute(
                    """
                    INSERT INTO feedback(timestamp_utc, latitude, longitude, category, description, rating)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        lat,
                        lon,
                        category,
                        description.strip(),
                        int(rating),
                    ),
                )
                con.commit()
            st.success("Thank you. Your feedback has been recorded.")
        except Exception as exc:
            st.error(f"Could not save feedback: {exc}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
init_db()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("Swaraksha could not load its dataset.")
    st.code(str(exc))
    st.stop()

# Minimal navigation: this is a safety-first application, not a classroom dashboard.
PAGES = [
    "🏠 Home",
    "📍 Check an Area",
    "🆘 Help Nearby",
    "🗺️ Live GIS Map",
    "👩 Women Safety",
    "👧 Child Safety",
    "🚨 Helplines",
    "👥 Community Feedback",
]

with st.sidebar:
    st.markdown("# 🛡️ Swaraksha")
    st.caption("Know the Pattern. Find the Help.")
    page = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.get("page", "🏠 Home")))
    st.session_state["page"] = page

    st.divider()
    st.markdown("### 📍 Location")
    if "lat" in st.session_state:
        st.success("Location selected")
        st.caption(f"{st.session_state['lat']:.5f}, {st.session_state['lon']:.5f}")
    else:
        st.caption("No location selected yet.")

    st.divider()
    st.markdown("### ⚠️ Remember")
    st.caption("Historical crime patterns do not guarantee present or future safety.")

if page == "🏠 Home":
    home_page(df)
elif page == "📍 Check an Area":
    check_area_page(df)
elif page == "🆘 Help Nearby":
    help_nearby_page(df)
elif page == "🗺️ Live GIS Map":
    map_page(df)
elif page == "👩 Women Safety":
    women_page(df)
elif page == "👧 Child Safety":
    child_page(df)
elif page == "🚨 Helplines":
    helpline_page()
elif page == "👥 Community Feedback":
    feedback_page()
