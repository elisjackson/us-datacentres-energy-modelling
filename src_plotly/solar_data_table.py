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
        Output("solar-data-table", "children"),
        Input("pv-location-data", "data"),
    )
    def update_solar_data_table(pv_location_data):

        key_mappings = {
            "ssrd": "GHI (W/m²)",
            "fdir": "DNI (W/m²)",
        }

        # template empty dataframe
        empty_df = pd.DataFrame({"Value": [""] * 3}, index=["GHI (W/m²)", "DNI (W/m²)", "Test"])

        if not pv_location_data:
            return dbc.Table.from_dataframe(empty_df, index=True, index_label="Property")

        df = pd.DataFrame([pv_location_data])
        df = df.rename(columns=key_mappings)
        df = df[[c for c in ["GHI (W/m²)", "DNI (W/m²)"] if c in df.columns]]

        if df.empty:
            return dbc.Table.from_dataframe(empty_df, index=True, index_label="Property")

        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].round(2)
        df = df.T
        df.columns = ["Value"]
        df.loc["Test", "Value"] = round(random.random(), 2)

        return dbc.Table.from_dataframe(df, index=True, index_label="Property")

solar_data_table_footer = html.P(
    ["GHI: Global Horizontal Irradiance", html.Br(), "DNI: Direct Normal Irradiance"],
    className="text-muted-small",
)