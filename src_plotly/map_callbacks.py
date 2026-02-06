"""
Map-related callbacks and helpers for the Dash app.
Builds choropleth map from GeoJSON, handles country/radio updates, and click highlight.
"""

import json
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

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


def _colorscale_to_rgba(colorscale_name, alpha=0.9):
    """Return a colorscale with the same colors as the built-in scale but with given alpha (0–1)."""
    import re
    raw = px.colors.get_colorscale(colorscale_name)
    out = []
    for pos, color in raw:
        if isinstance(color, str) and color.startswith("rgb("):
            match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color)
            if match:
                r, g, b = match.groups()
                out.append([pos, f"rgba({r},{g},{b},{alpha})"])
            else:
                out.append([pos, color])
        elif isinstance(color, str) and color.startswith("#"):
            from plotly.colors import hex_to_rgb
            from _plotly_utils.colors import unconvert_from_RGB_255
            t = hex_to_rgb(color)
            t = unconvert_from_RGB_255(t)
            r, g, b = (int(round(c * 255)) for c in t)
            out.append([pos, f"rgba({r},{g},{b},{alpha})"])
        else:
            out.append([pos, color])
    return out


# Cache for loaded GeoJSON + derived data; key = (filepath, color_on)
_geo_cache = {}
_gdf_cache = {}

def _get_gdf_data(country: str):
    """Load GDF once per country, cache result."""
    cache_key = country
    filepath = DATA_DIR / f"map_{country}_2025.geojson"

    if cache_key in _gdf_cache:
        return _gdf_cache[cache_key]

    with open(filepath, "r") as f:
        gdf = gpd.read_file(f)

    # check gdf is in EPSG:4326
    print(gdf.crs)
    if gdf.crs != "EPSG:4326":
        raise ValueError(f"GDF is not in EPSG:4326 for country {country}")

    _gdf_cache[cache_key] = gdf
    return gdf

def _get_geo_data(filepath, color_on, country):
    """Load GeoJSON once per (filepath, color_on), build df, center, and base figure; cache result."""
    if color_on == "wind_speed_100":
        colorscale = "Emrld_r"
        label = "Mean wind speed (m/s)"
        onshore_only_clickable = False
    elif color_on == "ssrd":
        colorscale = "solar"
        label = "Mean irradiation (W/m2)"
        onshore_only_clickable = True
    else:
        raise ValueError(f"Color on {color_on} not supported")

    cache_key = (filepath, color_on)
    if cache_key in _geo_cache:
        return _geo_cache[cache_key]

    gdf = _get_gdf_data(country)
    gdf = gdf.reset_index(names="id")
    geojson = json.loads(gdf.to_json())

    # gdf has an "onshore" column (boolean): True = onshore, False = offshore

    onshore_mask = gdf["onshore"].astype(bool)
    offshore_ids = gdf.loc[~onshore_mask, "id"].tolist()

    if onshore_only_clickable and offshore_ids:
        df = gdf.loc[onshore_mask].drop(columns="geometry")[["id", color_on, "onshore"]]
    else:
        df = gdf.drop(columns="geometry")[["id", color_on, "onshore"]]

    df["onshore_label"] = df["onshore"].apply(lambda x: "Onshore" if x else "Offshore")

    if country == "United States":
        center = {"lat": 39.19, "lon": -98.45}
        zoom = 2.5
    else:
        center = _center_from_geojson(geojson)
        zoom = 4

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
        hover_data={color_on: ":.2f", "id": False, "onshore_label": True},
        labels={color_on: label},
    )
    base_fig.update_traces(
        marker_line_width=0,
        zmin=df[color_on].min(),
        zmax=df[color_on].max(),
    )
    # Hover: show Onshore/Offshore only when onshore_only_clickable is False; always show metric
    hover_tpl = (
        "<b>%{customdata[2]}</b><br>" + label + "=%{z:.2f}<extra></extra>"
        if not onshore_only_clickable
        else label + "=%{z:.2f}<extra></extra>"
    )
    base_fig.update_traces(
        hovertemplate=hover_tpl,
        selector=dict(type="choroplethmap"),
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
        uirevision=color_on,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    base_fig.update_geos(visible=False, domain=dict(x=[0, 0.82], y=[0, 1]))

    if onshore_only_clickable and offshore_ids:
        offshore_z = gdf.loc[~onshore_mask, color_on].tolist()
        z_min, z_max = float(df[color_on].min()), float(df[color_on].max())
        offshore_colorscale = _colorscale_to_rgba(colorscale, alpha=0.25)
        offshore_trace = go.Choroplethmap(
            geojson=geojson,
            locations=offshore_ids,
            z=offshore_z,
            featureidkey="id",
            colorscale=offshore_colorscale,
            zmin=z_min,
            zmax=z_max,
            showscale=False,
            name="offshore",
            hoverinfo="skip",
            marker=dict(
                line=dict(color="rgba(0,0,0,0.05)", width=1),
            ),
        )
        base_fig.add_trace(offshore_trace)

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
            var curveNumber = clickData.points[0].curveNumber;
            if (curveNumber === 1) {
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
            var nTraces = figureData.data.length;
            var newData = JSON.parse(JSON.stringify(figureData.data.slice(0, nTraces)));
            newData.push(highlightTrace);
            var layout = figureData.layout || {};
            return { data: newData, layout: layout };
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
        if clickData and clickData.get("points"):
            if clickData["points"][0].get("curveNumber") == 1:
                return no_update
        return json.dumps(clickData, indent=2) if clickData else ""
