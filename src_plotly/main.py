from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

import src_plotly.map_callbacks as map_callbacks
import src_plotly.wind_profile as wind_profile

external_stylesheets = [dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
map_callbacks.register_callbacks(app)
wind_profile.register_callbacks(app)

# WSGI entry point for cloud (e.g. gunicorn src_plotly.main:server). Run from repo root.
server = app.server


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
                                dcc.Store(id="map-wind-max"),
                                dcc.Graph(
                                    id="map",
                                    style={"height": "600px"},
                                    config={"displayModeBar": False},
                                )
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
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
                                html.Div(
                                    [
                                        html.Span(
                                            "Solar data",
                                            className="h4"
                                        ),
                                        dbc.Table.from_dataframe(
                                            pd.DataFrame({"Solar data": ["1", "2", "3"]}),
                                            id="solar-data-table",
                                        ),
                                        # dcc.Store(id="hub-height", data=100),
                                        # dcc.Graph(
                                        #     id="wind-profile-graph",
                                        #     style={"height": "300px"},
                                        #     config={"displayModeBar": False},
                                        # ),
                                    ],
                                ),
                                html.Pre(id="click-data"),
                            ],
                            width=4,
                        ),
                    ],
                ),
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
