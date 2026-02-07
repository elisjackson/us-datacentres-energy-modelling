from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

import src_plotly.map_callbacks as map_callbacks
import src_plotly.wind_profile as wind_profile
import src_plotly.solar_data_table as solar_data_table
from src_plotly.solar_data_table import solar_data_table_footer

external_stylesheets = [dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
map_callbacks.register_callbacks(app)
wind_profile.register_callbacks(app)
solar_data_table.register_callbacks(app)

# WSGI entry point for cloud (e.g. gunicorn src_plotly.main:server). Run from repo root.
server = app.server


radioitems = html.Div(
    [
        dbc.Label("Select map layer"),
        dbc.RadioItems(
            options=[
                {"label": "Data Centre & PV location", "value": "PV"},
                {"label": "Wind location", "value": "Wind"},
            ],
            value="PV",
            id="radioitems-input",
            inline=True,
        ),
    ],
    className="mb-3",
)

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
                                            [
                                                html.Label("Select country", className="form-label"),
                                                dcc.Dropdown(
                                                    ['United Kingdom', 'United States'],
                                                    'United Kingdom',
                                                    id='country-dropdown'
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                html.Div("Some text", className="mb-3", id="map-helper-text"),
                                dcc.Store(id="figure-store"),
                                dcc.Store(id="map-wind-max"),
                                dcc.Store(id="last-country-store"),
                                dcc.Store(id="pv-click-store"),
                                dcc.Store(id="wind-click-store"),
                                dcc.Store(id="pv-location-data"),
                                dcc.Store(id="wind-location-data"),
                                dcc.Graph(
                                    id="map",
                                    style={"height": "600px"},
                                    config={"displayModeBar": False},
                                )
                            ],
                            width=8,
                            style={
                                "backgroundColor": "rgba(255,255,255,0.06)",
                                "borderRadius": "8px",
                                "padding": "1rem"
                            },
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Solar data",
                                            className="h4"
                                        ),
                                        html.Div(id="solar-data-table"),
                                        solar_data_table_footer,
                                    ],
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Wind data",
                                            className="h4"
                                        ),
                                        dcc.Store(id="hub-height", data=100),
                                        dcc.Graph(
                                            id="wind-profile-graph",
                                            style={"height": "300px"},
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                ),
                            ],
                            width=4,
                            className="d-flex flex-column justify-content-evenly gap-3",
                        ),
                    ],
                ),
            ],
            title="Select location",
        ),
        dbc.AccordionItem(
            [
                html.P("This is the content of the second section"),
                dbc.Button("Don't click me!", color="danger"),
            ],
            title="Optimiser parameters",
        ),
        dbc.AccordionItem(
            "This is the content of the third section",
            title="Results",
        ),
    ],
    active_item=0,
)

sidebar = html.Div(
    [
        html.H3("Datacentre energy optimiser", className="display-9"),
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
        accordion,
    ],
    className="content",
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

if __name__ == "__main__":
    # Local: run from repo root with python -m src_plotly.main (so src_plotly imports work)
    app.run(debug=True)
