# 🛡️ Swaraksha — Know the Pattern. Find the Help.

Location-first India safety web app built with Streamlit.

## Features
- India-wide OpenStreetMap/Leaflet map
- Indian address/locality search using Nominatim
- Browser/device geolocation using `streamlit-geolocation`
- Current mapped nearby police, hospitals, clinics, pharmacies, fire stations, legal offices and women/child support where OpenStreetMap has them
- District-level historical women-related crime indicator using the supplied CSV
- LOW / MODERATE / HIGH historical pattern labels
- Category-wise historical charts
- Time-aware future trend estimates (not guaranteed predictions)
- Emergency/support helplines
- Anonymous community feedback stored in SQLite

## Data
Put the supplied file at:

`data/raw/cleaned_community_safety_crime_2010_2022.csv`

The app checks that path first.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
Deploy the GitHub repository and set:
- Main file: `app.py`

No Google Maps key is required for the default OpenStreetMap/Nominatim/Overpass implementation.

## Important limitations
- The crime CSV is district-level. The app does not invent street-level crime hotspots.
- LOW/MODERATE/HIGH is a historical pattern indicator only.
- A missing current facility result does not prove that the facility does not exist.
- Public OSM services are shared services; cache is used to reduce repeated requests.
- Browser GPS requires the user's permission and a supported secure browser context.
- Do not treat the forecast as a guarantee of future crime.

## Helplines
The app displays Indian government emergency/support numbers such as 112, 181, 1098, 1930, 15100, 14416, 102, 101 and 100. Verify official numbers again before a public production launch.
