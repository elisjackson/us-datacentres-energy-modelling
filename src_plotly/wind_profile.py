import math
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, no_update


def calculate_wind_profile(v1, h1, z0):
    """
    On the ground, the wind is strongly braked by obstacles and surface roughness.
    High above the ground in the undisturbed air layers of the
    geostrophic wind (at approx. 5 km above ground) the wind is no longer
    influenced by the surface.
    
    Between these two extremes, wind speed changes with height.
    This phenomenon is called vertical wind shear.
    
    In flat terrain and with a neutrally stratisfied atmosphere,
    the logarithmic wind profile is a good estimation for the vertical wind shear:

    v2 = v1 * (np.log(h2 / z0) / np.log(h1 / z0))
    
    The reference wind speed v1 is measured at height h1.
    v2 is the wind speed at height h2. z0 is the roughness length (see table above).
    """

    heights = np.arange(0, 250, 10, dtype=float)
    # Avoid log(0): compute ratio only where height > 0; otherwise wind speed is 0
    ratio = np.zeros_like(heights)
    valid = heights > 0
    ratio[valid] = np.log(heights[valid] / z0) / np.log(h1 / z0)
    wind_speeds = v1 * ratio
    return heights, wind_speeds


def make_wind_profile_figure(heights, wind_speeds, x_max=None, hub_height=None):
    """
    Build a Plotly figure for wind profile (no .show()); for use in Dash.
    If x_max is set, x-axis range is fixed to [0, x_max].
    If hub_height is set, draw a horizontal line at that height and show wind speed there.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wind_speeds,
            y=heights,
            mode="lines",
            line=dict(color="#78b4a0", shape="spline"),
            marker=dict(color="#78b4a0"),
        )
    )
    xaxis = dict(
        tickfont=dict(color="#e0e0e0"),
        title_font=dict(color="#e0e0e0"),
        gridcolor="rgba(255,255,255,0.1)",
        zerolinecolor="rgba(255,255,255,0.2)",
        fixedrange=True,
    )
    if x_max is not None:
        xaxis["range"] = [0, x_max]
    layout_kw = dict(
        title="Wind profile (log law)",
        xaxis_title="Wind speed (m/s)",
        yaxis_title="Height (m)",
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        title_font=dict(color="#e0e0e0"),
        xaxis=xaxis,
        yaxis=dict(
            tickfont=dict(color="#e0e0e0"),
            title_font=dict(color="#e0e0e0"),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
            fixedrange=True,
        ),
        dragmode=False,
    )
    if hub_height is not None and heights.size > 0 and wind_speeds.size > 0:
        x_range_max = x_max if x_max is not None else float(np.max(wind_speeds))
        v_hub = float(np.interp(hub_height, heights, wind_speeds))
        layout_kw["shapes"] = [
            dict(
                type="line",
                x0=0,
                x1=x_range_max,
                y0=hub_height,
                y1=hub_height,
                line=dict(dash="dash", color="#e0e0e0", width=1.5),
            )
        ]
        layout_kw["annotations"] = [
            dict(
                x=v_hub,
                y=hub_height,
                text=f"{v_hub:.1f} m/s at {hub_height:.0f} m",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=0,
                font=dict(color="#e0e0e0", size=11),
                bgcolor="rgba(0,0,0,0.5)",
            )
        ]
    fig.update_layout(**layout_kw)
    return fig


def register_callbacks(app):
    """
    Register Dash callbacks that use this module (e.g. wind profile from map click).
    Call this from main.py after creating the app: wind_profile.register_callbacks(app).
    """
    @app.callback(
        Output("wind-profile-graph", "figure"),
        Input("map", "clickData"),
        Input("hub-height", "data"),
        State("map-wind-max", "data"),
    )
    def update_wind_profile(click_data, hub_height, map_wind_max):
        hub_height = 100 if hub_height is None else hub_height
        if not click_data:
            return go.Figure().update_layout(
                title="Wind profile (log law)",
                annotations=[
                    dict(
                        text="Click a map cell",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color="#e0e0e0", size=14),
                    )
                ],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
            )
        try:
            v1 = click_data["points"][0].get("z")
            v1 = 5.0 if v1 is None else float(v1)
        except (KeyError, IndexError, TypeError):
            v1 = 5.0
        h1 = 100.0  # reference height (m), e.g. ERA5 100 m
        z0 = 0.1  # roughness length (m)
        heights, wind_speeds = calculate_wind_profile(v1, h1, z0)
        # x-axis: at least ceil(map_wind_max)+1, or ceil(max profile) so 250 m speed is never clipped
        base_max = (math.ceil(map_wind_max) + 1) if map_wind_max is not None else 11
        profile_max = math.ceil(float(np.max(wind_speeds))) if wind_speeds.size else base_max
        x_max = max(base_max, profile_max)
        return make_wind_profile_figure(
            heights, wind_speeds, x_max=x_max, hub_height=hub_height
        )

    @app.callback(
        Output("hub-height", "data"),
        Input("wind-profile-graph", "clickData"),
        prevent_initial_call=True,
    )
    def set_hub_height_from_graph_click(click_data):
        """When user clicks on the wind profile, set hub height to the clicked y (height in m)."""
        if not click_data or not click_data.get("points"):
            return no_update
        try:
            y = click_data["points"][0].get("y")
            if y is None:
                return no_update
            h = round(float(y) / 10) * 10  # round to nearest 10 m
            h = max(10, min(250, h))
            return h
        except (KeyError, TypeError, ValueError):
            return no_update

