from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

import src_plotly.map_callbacks as map_callbacks
import src_plotly.wind_profile as wind_profile
import src_plotly.solar_data_table as solar_data_table
import src_plotly.concept_form as concept_form
from src_plotly.solar_data_table import solar_data_table_footer
from src_plotly.concept_form import concept_form_layout
from src_plotly.results_accordion import results_layout, register_callbacks as results_register_callbacks

external_stylesheets = [
    dbc.themes.DARKLY,
    dbc.icons.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap",
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
map_callbacks.register_callbacks(app)
wind_profile.register_callbacks(app)
solar_data_table.register_callbacks(app)
concept_form.register_callbacks(app)
results_register_callbacks(app)

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
                                dcc.Store(id="pv-latlon-store"),
                                dcc.Store(id="wind-latlon-store"),
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
            item_id="accordion-location",
        ),
        dbc.AccordionItem(
            [concept_form_layout()],
            title="Optimiser parameters",
            item_id="accordion-parameters",
        ),
        dbc.AccordionItem(
            [results_layout()],
            title="Results",
            item_id="accordion-results",
        ),
    ],
    id="main-accordion",
    flush=True,
    active_item="accordion-location",
)

# TODO - replace links
GITHUB_URL = "https://github.com"
LINKEDIN_URL = "https://linkedin.com"

LINKEDIN_ICON_SVG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ecf0f1'%3E"
    "%3Cpath d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/%3E"
    "%3C/svg%3E"
)

header_links = html.Div(
    [
        html.A(
            href=GITHUB_URL,
            target="_blank",
            rel="noopener noreferrer",
            children=html.Img(
                src="https://cdn.simpleicons.org/github/ecf0f1",
                alt="GitHub",
                className="header-icon",
            ),
        ),
        html.A(
            href=LINKEDIN_URL,
            target="_blank",
            rel="noopener noreferrer",
            children=html.Img(
                src=LINKEDIN_ICON_SVG,
                alt="LinkedIn",
                className="header-icon",
            ),
        ),
    ],
    className="header-links",
)

app_header = html.Div(
    [
        html.H1("Datacentre energy optimiser", className="app-header-title"),
        header_links,
    ],
    id="app-header",
    className="app-header",
)

main_content = html.Div(
    [
        accordion,
    ],
    className="content",
)

app.layout = dbc.Container(
    html.Div(
        [
            app_header,
            main_content,
        ],
        className="app-layout",
    ),
    fluid=True,
    className="app-container",
)

if __name__ == "__main__":
    # Local: run from repo root with python -m src_plotly.main (so src_plotly imports work)
    app.run(debug=True)
