# About

A Plotly Dash app and energy model for off-grid US data centres.

Visit my GitHub pages: [https://elisjackson.github.io/](elisjackson.github.io)

# Installation (using uv)

1. `uv venv`
2. `uv sync`

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

# Deployment (GCP Cloud Run)

The app is deployed at [https://datacentres.eliswyn.com](https://datacentres.eliswyn.com).

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated (`gcloud auth login`)
- GCP project `us-energy-data` with the following APIs enabled:
  - Cloud Build
  - Cloud Run
  - Artifact Registry

## Environment variables

Set the required environment variables on the Cloud Run service:

```bash
gcloud run services update us-datacentres --region us-central1 \
    --set-env-vars "OPTIMISER_API_URL=...,AWS_ACCESS_KEY_ID=...,AWS_SECRET_ACCESS_KEY=...,AWS_REGION=eu-west-2"
```

## Build and deploy

Run the deploy script from the repo root:

```powershell
.\deploy.ps1
```

This builds the Docker image via Cloud Build, pushes it to Artifact Registry, and deploys it to Cloud Run.