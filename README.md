# About

A simple Streamlit dashboard and energy model for off-grid US datacentres.

Visit my GitHub pages: [https://elisjackson.github.io/](elisjackson.github.io)

# Installation (using uv)

`uv venv`
`uv sync`

# Run the dashboard (Streamlit app)

`uv run streamlit run src/streamlit_dashboard.py`

# Energy model

The Streamlit app calls the `src/energy_model` module, using front end inputs. The module can also be executed stand-alone. To execute:
`uv run streamlit run src/energy_model/calculate_energy.py`

# Hex cell generator

The `src/generate_hex_cell_data` module downloads and processes US wind speed and Global Tilted Irradiance (GTI) data, and outputs two JSON files (to `src/hex_cell_data/`), that is used by the Streamlit app. It is not required to execute this module if you have cloned the repository as the files have been pre-generetaed.

The files saved to `src/hex_cell_data/` include:
- (GEOJSON) US wind speed and GTI data, with the US dividied into hex cells
- (JSON) A lookup US states -> hex cell IDs

To execute:
`uv run streamlit run src/generate_hex_cell_data/generate_data.py`

# Docs and assumptions

For documentation and assumptions, see the `docs` folder.