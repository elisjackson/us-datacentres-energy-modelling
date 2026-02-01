"""
Map-related callbacks and helpers for the Dash app.
Builds choropleth map from GeoJSON, handles country/radio updates, and click highlight.
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State

# GeoJSON dir: use DASH_DATA_DIR if set (e.g. in Plotly Cloud), else repo/data/processed
DATA_DIR = Path(
    os.environ.get(
        "DASH_DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "processed"),
    )
)


def _center_from_geojson(geojson):
    """Precompute (lat, lon) centre from GeoJSON feature geometries."""
    lons, lats = [], []
    for feat in geojson.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        coords = geom.get("coordinates")
        if geom["type"] == "Polygon":
            rings = [coords[0]] if coords else []
        elif geom["type"] == "MultiPolygon":
            rings = [p[0] for p in coords] if coords else []
        else:
            continue
        for ring in rings:
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
    if not lons:
        return None
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    return {"lat": center_lat, "lon": center_lon}


# Cache for loaded GeoJSON + derived data; key = (filepath, color_on)
_geo_cache = {}


def _get_geo_data(filepath, color_on, country):
    """Load GeoJSON once per (filepath, color_on), build df, center, and base figure (no highlight); cache result."""
    cache_key = (filepath, color_on)
    if cache_key in _geo_cache:
        return _geo_cache[cache_key]
    with open(filepath, "r") as f:
        geojson = json.load(f)
    features = geojson["features"]
    for i, feat in enumerate(features):
        feat["id"] = i
    df = pd.DataFrame(
        {
            "id": range(len(features)),
            color_on: [f["properties"][color_on] for f in features],
        }
    )

    if country == "United States":
        center = {"lat": 39.19, "lon": -98.45}
        zoom = 2.5
    else:
        center = _center_from_geojson(geojson)
        zoom = 4

    if color_on == "wind_speed_100":
        colorscale = "Emrld_r"
        label = "Mean wind speed (m/s)"
    elif color_on == "ssrd":
        colorscale = "solar"
        label = "Mean irradiation (W/m2)"
    else:
        raise ValueError(f"Color on {color_on} not supported")

    base_fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="id",
        color=color_on,
        color_continuous_scale=colorscale,
        featureidkey="id",
        opacity=0.5,
        center=center,
        zoom=zoom,
        hover_data={color_on: ":.2f", "id": False},
        labels={color_on: label},
    )
    base_fig.update_traces(
        marker_line_width=0,
        zmin=df[color_on].min(),
        zmax=df[color_on].max(),
    )
    base_fig.update_traces(
        colorbar=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            tickfont=dict(color="#e0e0e0"),
            title=dict(font=dict(color="#e0e0e0")),
        ),
        selector=dict(type="choroplethmap"),
    )
    base_fig.update_layout(
        margin=dict(r=0, t=0, l=0, b=0),
        uirevision="map",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    base_fig.update_geos(visible=False, domain=dict(x=[0, 0.82], y=[0, 1]))
    _geo_cache[cache_key] = {
        "geojson": geojson,
        "df": df,
        "center": center,
        "base_figure": base_fig,
    }
    return _geo_cache[cache_key]


def make_base_figure(radio_selection, country):
    """Base choropleth only (no highlight). Returns (figure, max_wind_speed) from the map data."""
    if country == "United Kingdom":
        filepath = DATA_DIR / f"map_{country}_2025.geojson"
    elif country == "United States":
        filepath = DATA_DIR / f"map_{country}_2025_01_01.geojson"
    else:
        raise ValueError(f"Country {country} not supported")
    filepath = str(filepath)
    if radio_selection == "Wind":
        color_on = "wind_speed_100"
    elif radio_selection == "PV":
        color_on = "ssrd"
    else:
        raise ValueError(f"Radio selection {radio_selection} not supported")
    geo_data = _get_geo_data(filepath, color_on, country)
    base_fig = geo_data["base_figure"]
    max_wind_speed = float(geo_data["df"][color_on].max())
    return go.Figure(base_fig), max_wind_speed


def register_callbacks(app):
    """Register map-related Dash callbacks. Call from main after creating the app."""
    @app.callback(
        [
            Output("map", "figure"),
            Output("figure-store", "data"),
            Output("map-wind-max", "data"),
        ],
        Input("radioitems-input", "value"),
        Input("country-dropdown", "value"),
    )
    def update_map_and_store(radio_selection, country):
        """Build base figure when country/radio change; store it and max wind for profile axis."""
        fig, max_wind = make_base_figure(radio_selection, country)
        return fig, fig.to_dict(), max_wind

    app.clientside_callback(
        """
        function(clickData, figureData) {
            if (!clickData || !figureData || !figureData.data || !figureData.data[0]) {
                return window.dash_clientside.no_update;
            }
            var selectedId = clickData.points[0].location;
            var baseTrace = figureData.data[0];
            var geo = baseTrace.geojson;
            if (!geo || !geo.features || !geo.features[selectedId]) {
                return window.dash_clientside.no_update;
            }
            var feature = geo.features[selectedId];
            var value = baseTrace.z && baseTrace.z[selectedId] != null ? baseTrace.z[selectedId] : 0;
            var highlightGeojson = { type: geo.type, features: [feature] };
            var highlightTrace = {
                type: 'choroplethmap',
                geojson: highlightGeojson,
                locations: [selectedId],
                z: [value],
                featureidkey: 'id',
                marker: { opacity: 1, line: { width: 1, color: '#282828' } },
                showscale: false,
                hoverinfo: 'skip',
                colorscale: [[0, 'rgba(120,180,160,0.5)'], [1, 'rgba(120,180,160,0.5)']]
            };
            if (baseTrace.zmin != null) highlightTrace.zmin = baseTrace.zmin;
            if (baseTrace.zmax != null) highlightTrace.zmax = baseTrace.zmax;
            if (baseTrace.zauto === false) highlightTrace.zauto = false;
            return { data: [baseTrace, highlightTrace], layout: figureData.layout };
        }
        """,
        Output("map", "figure", allow_duplicate=True),
        Input("map", "clickData"),
        State("figure-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("click-data", "children"),
        Input("map", "clickData"),
    )
    def display_click_data(clickData):
        return json.dumps(clickData, indent=2)
