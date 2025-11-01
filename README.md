# About

# Installation

`pip -m venv .venv`
`pip install -r requirements.txt`

# Data preparation

1. `generate_hex_cell_data\wind_and_pv_data.py` generates wind speed and PV GTI data, and outputs a hex cell representation (geojson file) of the US

# Streamlit app

`.venv\Scripts\activate`
`streamlit run streamlit_dashboard.py`