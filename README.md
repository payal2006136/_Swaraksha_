# 🛡️ SWARAKSHA — Know the Pattern. Find the Help.

Location-first India community-safety Streamlit application.

## Main features
- Search Indian city, district, locality, landmark or address using live geocoding.
- Browser/device current-location support after permission.
- India-wide OpenStreetMap/Leaflet map.
- Current mapped nearby support within a selectable radius: police, hospitals, clinics/health centres, pharmacies, fire stations, legal services and women/child support where mapped.
- Map markers and a visible radius circle. The circle colour represents the **district-level historical pattern**, not a street-level crime hotspot.
- Historical women-related crime indicators: LOW / MODERATE / HIGH for available categories.
- Plotly historical trend and category comparison charts.
- Category-specific time-aware future trend estimates for the selected district.
- Anonymous SQLite community feedback.
- Emergency and support helplines.

## Data limitation that must remain visible
The supplied crime dataset is district-level. Therefore, if a user searches a locality such as Vashi, the historical analysis is for the matched district (for example, Thane) and is **not** a claim about the exact street or locality. The application never fabricates a street-level crime result.

Historical crime patterns do not guarantee present or future safety.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
Deploy from GitHub with `app.py` as the main file. The CSV must remain at:

`data/raw/cleaned_community_safety_crime_2010_2022.csv`

## Live geographic services
- OpenStreetMap tiles for the interactive map.
- Nominatim for Indian location search/geocoding.
- Overpass API for currently mapped nearby facilities.

These public services can be rate-limited or temporarily unavailable. The application does not invent facilities when a service fails.
