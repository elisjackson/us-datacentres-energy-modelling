from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import json
import pandas as pd

external_stylesheets = [dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

@app.callback(
    [Output("map", "figure"), Output("figure-store", "data")],
    Input("radioitems-input", "value"),
    Input("country-dropdown", "value"),
)
def update_map_and_store(radio_selection, country):
    """Build base figure (no highlight) when country/radio change; store it for clientside clicks."""
    fig = make_base_figure(radio_selection, country)
    return fig, fig.to_dict()


def make_base_figure(radio_selection, country):
    """Base choropleth only (no highlight). Used for initial display and for figure-store."""
    if country == "United Kingdom":
        filepath = rf"C:\Users\Elis\repos\us-datacentres\data\processed\mean_wind_speed_{country}_2025.geojson"
    elif country == "United States":
        filepath = rf"C:\Users\Elis\repos\us-datacentres\data\processed\mean_wind_speed_{country}_2025_01_01.geojson"
    else:
        raise ValueError(f"Country {country} not supported")
    # filepath = rf"C:\Users\Elis\repos\us-datacentres\data\processed\mean_wind_speed_{country}_2025_01_01.geojson"
    color_on = "mean_wind_speed"
    geo_data = _get_geo_data(filepath, color_on, country)
    base_fig = geo_data["base_figure"]
    return go.Figure(base_fig)


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


# Cache for loaded GeoJSON + derived data; key = filepath (or country when filepath varies).
# Avoids reloading and recomputing on every click when only selection changes.
_geo_cache = {}


def _get_geo_data(filepath, color_on, country):
    """Load GeoJSON once per filepath, build df, center, and base figure (no highlight); cache result."""
    if filepath in _geo_cache:
        return _geo_cache[filepath]
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
    # Build base figure once (expensive); on click we only copy it and add the highlight trace.
    base_fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="id",
        color=color_on,
        color_continuous_scale="Emrld",
        featureidkey="id",
        opacity=0.5,
        center=center,
        zoom=zoom,
        #         hover_name=None,
        hover_data={color_on: ":.2f", "id": False},
        labels={color_on: "Mean wind speed (m/s)"},
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
    base_fig.update_geos(visible=False)
    _geo_cache[filepath] = {
        "geojson": geojson,
        "df": df,
        "center": center,
        "base_figure": base_fig,
    }
    return _geo_cache[filepath]


def make_choropleth_v0(radio_selection, click_data, country):
    
    if click_data:
        # point_index = click_data["points"][0]["pointNumber"]
        selected_id = click_data["points"][0]["location"]
        # print(point_index)
        # print(selected_id)
    else:
        selected_id = "121-La Pointe-aux-Prairies"

    df = px.data.election()
    geojson = px.data.election_geojson()

    # get a subset of the geojson
    features = geojson["features"]
    highlighted_geojson = {}
    highlighted_geojson["type"] = geojson["type"]
    highlighted_geojson["features"] = []
    for f in features:
        if f["properties"]["district"] == selected_id:
            highlighted_geojson["features"].append(f)

    if radio_selection == "Wind":
        color_on = "Joly"
    elif radio_selection == "PV":
        color_on = "Coderre"

    fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="district",
        color=color_on,
        featureidkey="properties.district",
        center={"lat": 45.5517, "lon": -73.7073},
        zoom=9,
        range_color=[0, 6500],
        opacity=0.5
    )
    fig.update_layout(margin=dict(r=0, t=0, l=0, b=0))

    fig.add_trace(
        px.choropleth_map(
            df,
            geojson=highlighted_geojson,
            locations="district",
            color=color_on,
            featureidkey="properties.district",
            range_color=[0, 6500],
            opacity=1
        ).data[0]
    )

    return fig

radioitems = html.Div(
    [
        dbc.Label("Choose one"),
        dbc.RadioItems(
            options=[
                {"label": "Wind", "value": "Wind"},
                {"label": "PV", "value": "PV"},
            ],
            value="Wind",
            id="radioitems-input",
            inline=True,
        ),
    ],
    className="mb-3",
)

items = [
    dbc.DropdownMenuItem("Item 1"),
    dbc.DropdownMenuItem("Item 2"),
    dbc.DropdownMenuItem("Item 3"),
]

accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(radioitems),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                ['United Kingdom', 'United States'],
                                                'United Kingdom',
                                                id='country-dropdown'
                                            )
                                        ),
                                    ],
                                ),
                                dcc.Store(id="figure-store"),
                                dcc.Graph(
                                    id="map",
                                    style={"height": "600px"}
                                )
                            ],
                            width=8,
                        ),
                        dbc.Col(html.Pre(id="click-data"), width=4)
                    ]
                )
            ],
            title="Item 1",
        ),
        dbc.AccordionItem(
            [
                html.P("This is the content of the second section"),
                dbc.Button("Don't click me!", color="danger"),
            ],
            title="Item 2",
        ),
        dbc.AccordionItem(
            "This is the content of the third section",
            title="Item 3",
        ),
    ],
    active_item=0,
)

sidebar = html.Div(
    [
        html.H2("My App", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="#"),
                dbc.NavLink("Analytics", href="#"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

main_content = html.Div(
    [
        html.H1("Hello World"),
        accordion,
    ],
    className="content",
)

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

app.layout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, width=2, className="sidebar-col"),
            dbc.Col(main_content, width=10),
        ],
        className="layout-row g-0",
    ),
    fluid=True,
)

@app.callback(
    Output('click-data', 'children'),
    Input('map', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


if __name__ == "__main__":
    app.run(debug=True)
