"""
Map-related callbacks and helpers for the Dash app.
Builds choropleth map from GeoJSON, handles country/radio updates, and click highlight.
"""

import copy
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
    if gdf.crs != "EPSG:4326":
        raise ValueError(f"GDF is not in EPSG:4326 for country {country}")

    _gdf_cache[cache_key] = gdf
    return gdf


def _gdf_row_from_click(gdf, click_data):
    """Extract the clicked row from gdf with centroid coordinates. Returns dict or None. Use with clickData points[0].location."""
    if not click_data or not click_data.get("points"):
        return None
    loc = click_data["points"][0].get("location")
    if loc is None:
        return None
    row = gdf.loc[gdf["id"] == loc]
    if row.empty:
        return None
    
    # Get centroid coordinates before dropping geometry
    centroid = row.geometry.iloc[0].centroid
    result = row.drop(columns="geometry").iloc[0].to_dict()
    result["lat"] = centroid.y
    result["lon"] = centroid.x
    
    return result


def _get_geo_data(filepath, color_on, country):
    """Load GeoJSON once per (filepath, color_on), build df, center, and base figure; cache result."""
    if color_on == "wind_speed_100":
        colorscale = "Emrld_r"
        label = "Mean 100m wind speed (m/s)"
        onshore_only_clickable = False
    elif color_on == "ssrd":
        colorscale = "solar"
        label = "Mean irradiation (W/m²)"
        onshore_only_clickable = True
    else:
        raise ValueError(f"Color on {color_on} not supported")

    cache_key = (filepath, color_on)
    if cache_key in _geo_cache:
        cached = _geo_cache[cache_key]
        if "base_figure_dict" not in cached:
            cached["base_figure_dict"] = cached["base_figure"].to_dict()
        return cached

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

    base_figure_dict = base_fig.to_dict()
    _geo_cache[cache_key] = {
        "geojson": geojson,
        "df": df,
        "center": center,
        "base_figure": base_fig,
        "base_figure_dict": base_figure_dict,
    }
    return _geo_cache[cache_key]


def _add_highlight_trace(fig, geojson, selected_id, z_value, zmin, zmax, color_on):
    """Add a highlight trace for the selected polygon to the figure."""
    if not geojson or "features" not in geojson:
        return
    features = geojson["features"]
    feature = None
    if isinstance(selected_id, int) and 0 <= selected_id < len(features):
        feature = features[selected_id]
    else:
        for f in features:
            if f.get("id") == selected_id:
                feature = f
                break
    if feature is None:
        return
    highlight_geojson = {"type": geojson["type"], "features": [feature]}
    wind_color = "rgba(120,180,160,0.5)"
    pv_color = "rgba(255,180,80,0.5)"
    highlight_color = pv_color if color_on == "ssrd" else wind_color
    highlight_trace = go.Choroplethmap(
        geojson=highlight_geojson,
        locations=[selected_id],
        z=[z_value],
        featureidkey="id",
        colorscale=[[0, highlight_color], [1, highlight_color]],
        zmin=zmin,
        zmax=zmax,
        showscale=False,
        hoverinfo="skip",
        marker=dict(opacity=1, line=dict(width=1, color="#282828")),
        name="map_highlight",
    )
    fig.add_trace(highlight_trace)


def _add_highlight_trace_to_fig_dict(fig_dict, geojson, selected_id, z_value, zmin, zmax, color_on):
    """Append a highlight trace dict to fig_dict['data']. Mutates fig_dict in place."""
    if not geojson or "features" not in geojson:
        return
    features = geojson["features"]
    feature = None
    if isinstance(selected_id, int) and 0 <= selected_id < len(features):
        feature = features[selected_id]
    else:
        for f in features:
            if f.get("id") == selected_id:
                feature = f
                break
    if feature is None:
        return
    highlight_geojson = {"type": geojson["type"], "features": [copy.deepcopy(feature)]}
    wind_color = "rgba(120,180,160,0.5)"
    pv_color = "rgba(255,180,80,0.5)"
    highlight_color = pv_color if color_on == "ssrd" else wind_color
    trace_dict = {
        "type": "choroplethmap",
        "geojson": highlight_geojson,
        "locations": [selected_id],
        "z": [z_value],
        "featureidkey": "id",
        "colorscale": [[0, highlight_color], [1, highlight_color]],
        "zmin": zmin,
        "zmax": zmax,
        "showscale": False,
        "hoverinfo": "skip",
        "marker": {"opacity": 1, "line": {"width": 1, "color": "#282828"}},
        "name": "map_highlight",
    }
    fig_dict.setdefault("data", []).append(trace_dict)


def _apply_relayout_to_fig_dict(fig_dict, relayout_data, country_changed):
    """Apply center/zoom from relayout_data to a figure dict. Returns a new dict (does not mutate input)."""
    if country_changed or not relayout_data:
        return copy.deepcopy(fig_dict)
    # Shallow copy top level; deep-copy only layout so we don't copy heavy trace data (GeoJSON).
    out = dict(fig_dict)
    out["layout"] = copy.deepcopy(fig_dict.get("layout", {}))
    layout = out["layout"]
    for subplot in ("mapbox", "geo", "map"):
        center = relayout_data.get(f"{subplot}.center")
        zoom = relayout_data.get(f"{subplot}.zoom")
        if center is not None and zoom is not None:
            layout.setdefault(subplot, {}).update(center=center, zoom=zoom)
            return out
    return out


def make_base_figure(radio_selection, country, geo_data=None):
    """Resolve geo_data and max_wind_speed for the current radio/country. Returns (geo_data, max_wind_speed)."""
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

    geo_data = geo_data or _get_geo_data(filepath, color_on, country)
    geo_wind = _get_geo_data(filepath, "wind_speed_100", country)
    max_wind_speed = float(geo_wind["df"]["wind_speed_100"].max())
    return geo_data, max_wind_speed


def prewarm_geo_cache():
    """Load and cache geo data for all (country, metric) combinations at startup so the first user gets a hot cache."""
    for country in ["United Kingdom"]:
        if country == "United Kingdom":
            filepath = str(DATA_DIR / f"map_{country}_2025.geojson")
        else:
            filepath = str(DATA_DIR / f"map_{country}_2025_01_01.geojson")
        for color_on in ("wind_speed_100", "ssrd"):
            _get_geo_data(filepath, color_on, country)


def register_callbacks(app):
    """Register map-related Dash callbacks. Call from main after creating the app."""

    @app.callback(
        [
            Output("map", "figure"),
            Output("figure-store", "data"),
            Output("map-wind-max", "data"),
            Output("last-country-store", "data"),
            Output("pv-click-store", "data", allow_duplicate=True),
            Output("wind-click-store", "data", allow_duplicate=True),
        ],
        Input("radioitems-input", "value"),
        # Input("country-dropdown", "value"),
        State("map", "relayoutData"),
        State("last-country-store", "data"),
        State("pv-click-store", "data"),
        State("wind-click-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_map_and_store(
        radio_selection,
        # country,
        relayout_data,
        last_country,
        pv_click,
        wind_click,
    ):
        """Build base figure when country/radio change; preserve zoom/center when only layer changes."""
        country = "United Kingdom"
        geo_data, max_wind = make_base_figure(radio_selection, country)
        country_changed = (last_country is None) or (country != last_country)
        pv_out, wind_out = pv_click, wind_click
        if country_changed:
            pv_out, wind_out = None, None

        def _need_highlight():
            for stored_click in (pv_click, wind_click):
                if stored_click and stored_click.get("points"):
                    pt = stored_click["points"][0]
                    if pt.get("curveNumber") != 1:
                        return True
            return False

        if not _need_highlight():
            out_dict = _apply_relayout_to_fig_dict(
                geo_data["base_figure_dict"], relayout_data, country_changed
            )
            return out_dict, out_dict, max_wind, country, pv_out, wind_out

        out_dict = copy.deepcopy(geo_data["base_figure_dict"])
        if not country_changed and relayout_data:
            layout = out_dict.setdefault("layout", {})
            for subplot in ("mapbox", "geo", "map"):
                center = relayout_data.get(f"{subplot}.center")
                zoom = relayout_data.get(f"{subplot}.zoom")
                if center is not None and zoom is not None:
                    layout.setdefault(subplot, {}).update(center=center, zoom=zoom)
                    break
        filepath = str(DATA_DIR / f"map_{country}_2025.geojson")
        if country == "United States":
            filepath = str(DATA_DIR / f"map_{country}_2025_01_01.geojson")
        geo_pv = _get_geo_data(filepath, "ssrd", country)
        geo_wind = _get_geo_data(filepath, "wind_speed_100", country)
        for stored_click, gdata, color_on in [
            (pv_click, geo_pv, "ssrd"),
            (wind_click, geo_wind, "wind_speed_100"),
        ]:
            if stored_click and stored_click.get("points"):
                pt = stored_click["points"][0]
                if pt.get("curveNumber") != 1:
                    try:
                        loc = pt.get("location")
                        z_val = pt.get("z", 0) or 0
                        zmin = float(gdata["df"][color_on].min())
                        zmax = float(gdata["df"][color_on].max())
                        _add_highlight_trace_to_fig_dict(
                            out_dict, gdata["geojson"], loc, z_val, zmin, zmax, color_on
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
        return out_dict, out_dict, max_wind, country, pv_out, wind_out

    app.clientside_callback(
        """
        function(clickData, figureData, radioSelection, pvClick, windClick) {
            if (!clickData || !figureData || !figureData.data || !figureData.data[0]) {
                return window.dash_clientside.no_update;
            }
            var curveNumber = clickData.points[0].curveNumber;
            if (curveNumber === 1) {
                return window.dash_clientside.no_update;
            }
            var baseTrace = figureData.data[0];
            var geo = baseTrace.geojson;
            if (!geo || !geo.features) {
                return window.dash_clientside.no_update;
            }
            var windColor = 'rgba(120,180,160,0.5)';
            var pvColor = 'rgba(255,180,80,0.5)';
            var pvData = (radioSelection === 'PV') ? clickData : pvClick;
            var windData = (radioSelection === 'Wind') ? clickData : windClick;
            var baseTraces = figureData.data.filter(function(t) { return t.name !== 'map_highlight'; });
            var newData = JSON.parse(JSON.stringify(baseTraces));
            function addHighlight(clickObj, color) {
                if (!clickObj || typeof clickObj !== 'object' || !Array.isArray(clickObj.points) || !clickObj.points[0] || clickObj.points[0].curveNumber === 1) return;
                var loc = clickObj.points[0].location;
                if (!geo.features[loc]) return;
                var zVal = clickObj.points[0].z != null ? clickObj.points[0].z : 0;
                var feat = geo.features[loc];
                var hTrace = {
                    type: 'choroplethmap',
                    geojson: { type: geo.type, features: [JSON.parse(JSON.stringify(feat))] },
                    locations: [loc],
                    z: [zVal],
                    featureidkey: 'id',
                    marker: { opacity: 1, line: { width: 1, color: '#282828' } },
                    showscale: false,
                    hoverinfo: 'skip',
                    colorscale: [[0, color], [1, color]],
                    name: 'map_highlight'
                };
                if (typeof baseTrace.zmin === 'number') hTrace.zmin = baseTrace.zmin;
                if (typeof baseTrace.zmax === 'number') hTrace.zmax = baseTrace.zmax;
                if (baseTrace.zauto === false) hTrace.zauto = false;
                newData.push(JSON.parse(JSON.stringify(hTrace)));
            }
            addHighlight(pvData, pvColor);
            addHighlight(windData, windColor);
            var layout = figureData.layout ? JSON.parse(JSON.stringify(figureData.layout)) : {};
            return JSON.parse(JSON.stringify({ data: newData, layout: layout }));
        }
        """,
        Output("map", "figure", allow_duplicate=True),
        Input("map", "clickData"),
        [State("figure-store", "data"), State("radioitems-input", "value"),
         State("pv-click-store", "data"), State("wind-click-store", "data")],
        prevent_initial_call=True,
    )

    @app.callback(
        [Output("pv-click-store", "data"), Output("wind-click-store", "data")],
        Input("map", "clickData"),
        State("radioitems-input", "value"),
        State("pv-click-store", "data"),
        State("wind-click-store", "data"),
    )
    def save_click_to_store(click_data, radio_selection, pv_click, wind_click):
        """Save map click to the appropriate store (PV or Wind) based on current layer."""
        if not click_data or not click_data.get("points"):
            return no_update, no_update
        if click_data["points"][0].get("curveNumber") == 1:
            return no_update, no_update
        if radio_selection == "PV":
            return click_data, no_update
        else:
            return no_update, click_data

    @app.callback(
        Output("pv-location-data", "data"),
        Input("pv-click-store", "data"),
        # State("country-dropdown", "value"),
    )
    def store_pv_location_data(pv_click):
        """Store all GDF columns for the selected PV location."""
        country = "United Kingdom"
        if not pv_click or not country:
            return None
        gdf = _get_gdf_data(country).reset_index(names="id")
        return _gdf_row_from_click(gdf, pv_click)

    @app.callback(
        Output("wind-location-data", "data"),
        Input("wind-click-store", "data"),
        # State("country-dropdown", "value"),
    )
    def store_wind_location_data(wind_click):
        """Store all GDF columns for the selected Wind location."""
        country = "United Kingdom"
        if not wind_click or not country:
            return None
        gdf = _get_gdf_data(country).reset_index(names="id")
        return _gdf_row_from_click(gdf, wind_click)

    @app.callback(
        Output("wind-onshore-store", "data"),
        Input("wind-location-data", "data"),
    )
    def store_wind_onshore(wind_location_data):
        """Store whether the selected Wind location is onshore (True) or offshore (False). Used by wind profile and elsewhere."""
        if not wind_location_data or "onshore" not in wind_location_data:
            return None
        return bool(wind_location_data["onshore"])

    @app.callback(
        Output("wind-latlon-store", "data"),
        Input("wind-location-data", "data"),
    )
    def store_wind_latlon(wind_location_data):
        """Store lat/lon and onshore/offshore flag from the Wind location."""
        if not wind_location_data or "lat" not in wind_location_data or "lon" not in wind_location_data:
            return None
        result = {
            "lat": wind_location_data["lat"],
            "lon": wind_location_data["lon"],
        }
        if "onshore" in wind_location_data:
            result["onshore"] = bool(wind_location_data["onshore"])
        return result

    @app.callback(
        Output("map-helper-text", "children"),
        Input("radioitems-input", "value"),
    )
    def display_map_helper_text(radio_selection):
        if radio_selection == "Wind":
            return "Select a location for the wind farm. This may be different to the data centre location."
        elif radio_selection == "PV":
            return "Select a location for the data centre. Solar PV will be assumed to be co-located."
        else:
            return "Placeholder text"
