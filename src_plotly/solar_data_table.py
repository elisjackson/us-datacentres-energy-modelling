import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, html
import random

def register_callbacks(app):
    """
    Register Dash callbacks that use this module.
    Call this from main.py after creating the app: solar_data_table.register_callbacks(app).
    """
    @app.callback(
        Output("pv-latlon-store", "data"),
        Input("pv-location-data", "data"),
    )
    def store_pv_latlon(pv_location_data):
        """Store only the lat/lon coordinates from the PV location."""
        if not pv_location_data or "lat" not in pv_location_data or "lon" not in pv_location_data:
            return None
        return {
            "lat": pv_location_data["lat"],
            "lon": pv_location_data["lon"]
        }
    
    @app.callback(
        Output("solar-data-table", "children"),
        Input("pv-location-data", "data"),
    )
    def update_solar_data_table(pv_location_data):

        key_mappings = {
            "ssrd": "GHI (W/m²)",
            "fdir": "DNI (W/m²)",
            "lat": "Latitude",
            "lon": "Longitude",
        }

        # template empty dataframe
        empty_df = pd.DataFrame({"Value": [""] * 4}, index=["GHI (W/m²)", "DNI (W/m²)", "Latitude", "Longitude"])

        if not pv_location_data:
            return dbc.Table.from_dataframe(empty_df, index=True, index_label="Property")

        df = pd.DataFrame([pv_location_data])
        df = df.rename(columns=key_mappings)
        df = df[[c for c in ["GHI (W/m²)", "DNI (W/m²)", "Latitude", "Longitude"] if c in df.columns]]

        if df.empty:
            return dbc.Table.from_dataframe(empty_df, index=True, index_label="Property")

        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].round(4)
        df = df.T
        df.columns = ["Value"]

        return dbc.Table.from_dataframe(df, index=True, index_label="Property")

solar_data_table_footer = html.P(
    ["GHI: Global Horizontal Irradiance", html.Br(), "DNI: Direct Normal Irradiance"],
    className="text-muted-small",
)