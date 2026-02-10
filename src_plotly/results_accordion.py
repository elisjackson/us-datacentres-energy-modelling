import json
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

DEBUG_PRINT_TO_TERMINAL = True  # Set False to disable server-side print when Store updates

def create_card(header: str, value: float):
    return dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(
                [
                    html.H5(round(value), className="card-title"),
                ]
            )
        ]
    )


def results_layout():
    return html.Div(
        [
            dcc.Store(id="optimiser-results-data"),  # In-memory only; storage_type="local" can cause "Maximum update depth exceeded" with dependent callbacks
            dbc.Collapse(
                [
                    html.H6("Results Store (debug)", className="mt-3 mb-2"),
                    html.Div(id="optimiser-results-debug-content"),
                ],
                id="optimiser-results-debug-collapse",
                is_open=False,
            ),
            dbc.Button(
                "Show / hide results data (debug)",
                id="optimiser-results-debug-toggle",
                color="secondary",
                size="sm",
                outline=True,
                className="mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(create_card("Total Cost", 1000), id="total-cost-card"),
                    dbc.Col(create_card("Total Emissions", 10000), id="total-emissions-card"),
                ]
            ),
            dbc.Row(dbc.Col(html.Div("A single column"))),
        ],
        className="p-3",
        style={
            "backgroundColor": "rgba(255,255,255,0.06)",
            "borderRadius": "8px",
            "width": "100%",
        },
    )


def register_callbacks(app):
    """Register callbacks for the results accordion (debug display + any future results UI)."""

    @app.callback(
        [
            Output("total-cost-card", "children"),
            Output("total-emissions-card", "children"),
        ],
        Input("optimiser-results-data", "data"),
    )
    def _update_summary_cards(data):
        if data is None:
            return create_card("Total Cost", 0), create_card("Total Emissions", 0)
        return (
            create_card("Total Cost", data.get("total_cost", 0)),
            create_card("Total Emissions", data.get("total_emissions", 0)),
        )

    @app.callback(
        Output("optimiser-results-debug-content", "children"),
        Input("optimiser-results-data", "data"),
    )
    def _show_results_store_for_debugging(data):
        if DEBUG_PRINT_TO_TERMINAL and data is not None:
            print("[optimiser-results-data]", json.dumps(data, indent=2, default=str))
        if data is None:
            summary = "No results yet. Run the optimiser to populate."
            return html.Pre(summary, className="text-muted small")

        summary = "Keys: " + ", ".join(data.keys())
        try:
            full_json_text = json.dumps(data, indent=2, default=str)
        except (TypeError, ValueError):
            full_json_text = str(data)
        return html.Div(
            [
                html.Pre(summary, className="text-muted small"),
                html.Pre(full_json_text, style={"maxHeight": "400px", "overflow": "auto", "fontSize": "12px"})
            ]
        )

    @app.callback(
        Output("optimiser-results-debug-collapse", "is_open"),
        Input("optimiser-results-debug-toggle", "n_clicks"),
        State("optimiser-results-debug-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_debug_collapse(n_clicks, is_open):
        return not is_open