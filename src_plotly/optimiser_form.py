"""
Optimiser parameters form: 4 rows × (toggle | slider 0–100 | dropdown low/mid/high).
When a row's toggle is off, its slider and dropdown are disabled and greyed (values retained).
"""
from dash import dcc, html, Input, Output, State, ALL, no_update
import dash_bootstrap_components as dbc

from src_plotly.optimise import run_optimisation

N_ROWS = 4
ROW_IDS = list(range(N_ROWS))
ROW_LABELS = [
    "Data Centre Capacity",
    "Wind Farm Capacity",
    "Solar PV Capacity",
    "Grid Connection",
]

# Column widths (Bootstrap grid, must sum to 12): Parameter | Toggle | Level | Tier
W_PARAM, W_TOGGLE, W_LEVEL, W_TIER = 2, 1, 5, 4

# ID type prefixes so callbacks don't clash with other components
_TOGGLE = "optimiser-toggle"
_SLIDER = "optimiser-slider"
_SLIDER_VALUE = "optimiser-slider-value"
_DROPDOWN = "optimiser-dropdown"
_ROW_WRAPPER = "optimiser-row-wrapper"


def _make_row(row_id: int):
    return dbc.Row(
        [
            dbc.Col(
                html.Span(ROW_LABELS[row_id], className="text-nowrap"),
                width=W_PARAM,
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dbc.Switch(
                    id={"type": _TOGGLE, "index": row_id},
                    label="",
                    value=True,
                    className="mb-0",
                ),
                width=W_TOGGLE,
                className="d-flex align-items-center justify-content-center",
            ),
            dbc.Col(
                html.Div(
                    [
                        dbc.Input(
                            type="range",
                            id={"type": _SLIDER, "index": row_id},
                            min=0,
                            max=100,
                            step=1,
                            value=50,
                            className="form-range row-slider-input",
                            debounce=False,
                        ),
                        html.Span(
                            id={"type": _SLIDER_VALUE, "index": row_id},
                            className="row-slider-value ms-2 text-muted",
                            style={"fontSize": "0.9rem"},
                        ),
                    ],
                    className="d-flex align-items-center row-slider-wrapper",
                ),
                width=W_LEVEL,
                className="d-flex align-items-center justify-content-center",
            ),
            dbc.Col(
                html.Div(
                    dcc.Dropdown(
                        id={"type": _DROPDOWN, "index": row_id},
                        options=[
                            {"label": "Low", "value": "low"},
                            {"label": "Mid", "value": "mid"},
                            {"label": "High", "value": "high"},
                        ],
                        value="mid",
                        clearable=False,
                    ),
                    className="optimiser-tier-dropdown-wrapper w-100",
                ),
                width=W_TIER,
                className="d-flex align-items-center justify-content-center",
            ),
        ],
        className="mb-3 align-items-center row-controls",
        id={"type": _ROW_WRAPPER, "index": row_id},
    )


def optimiser_form_layout():
    """Build the optimiser parameters form (toggle | slider | dropdown per row)."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Strong("Parameter"), width=W_PARAM),
                    dbc.Col(html.Strong("Toggle"), width=W_TOGGLE, className="text-center"),
                    dbc.Col(html.Strong("Level (0–100)"), width=W_LEVEL),
                    dbc.Col(html.Strong("Tier"), width=W_TIER),
                ],
                className="mb-2 text-muted small",
            ),
            *[_make_row(i) for i in ROW_IDS],
            dbc.Button("Optimise", id="optimiser-optimise-button", color="primary", className="mt-3"),
            html.Div(id="optimiser-result", className="mt-2 text-muted small"),
            dcc.Store(id="optimiser-trigger-run"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Optimising")),
                    dbc.ModalBody(
                        [
                            dbc.Spinner(color="primary", size="sm", spinner_class_name="me-2"),
                            "Running optimisation…",
                        ],
                        className="d-flex align-items-center",
                    ),
                ],
                id="optimiser-loading-modal",
                is_open=False,
                centered=True,
                backdrop="static",
                keyboard=False,
            ),
        ],
        className="p-3",
        style={
            "backgroundColor": "rgba(255,255,255,0.06)",
            "borderRadius": "8px",
            "width": "100%",
        },
    )


def register_callbacks(app):
    """Register callbacks for the optimiser form (row enable/disable, slider value display)."""
    # Disable slider and dropdown when toggle is off; add row-disabled class
    @app.callback(
        [
            Output({"type": _SLIDER, "index": ALL}, "disabled"),
            Output({"type": _DROPDOWN, "index": ALL}, "disabled"),
            Output({"type": _ROW_WRAPPER, "index": ALL}, "className"),
        ],
        [Input({"type": _TOGGLE, "index": ALL}, "value")],
    )
    def _sync_row_state(toggles):
        disabled_list = [not (t is True) for t in toggles]
        row_class = [
            "mb-3 align-items-center row-controls" + (" row-disabled" if d else "")
            for d in disabled_list
        ]
        return disabled_list, disabled_list, row_class

    # Show current slider value next to each slider
    @app.callback(
        Output({"type": _SLIDER_VALUE, "index": ALL}, "children"),
        Input({"type": _SLIDER, "index": ALL}, "value"),
    )
    def _update_slider_display(values):
        return [str(v) if v is not None else "—" for v in values]

    # Optimise button: open loading modal and trigger run
    @app.callback(
        [
            Output("optimiser-loading-modal", "is_open"),
            Output("optimiser-trigger-run", "data"),
        ],
        Input("optimiser-optimise-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_loading_modal(n_clicks):
        return True, n_clicks

    # Run optimisation when triggered; close modal, show result, open Results accordion
    _RESULTS_ACCORDION_ITEM_ID = "accordion-results"  # item_id of Results accordion item (main.py)

    @app.callback(
        [
            Output("optimiser-result", "children"),
            Output("optimiser-loading-modal", "is_open", allow_duplicate=True),
            Output("optimiser-trigger-run", "data", allow_duplicate=True),
            Output("main-accordion", "active_item"),
            Output("optimiser-results-content", "children"),
        ],
        Input("optimiser-trigger-run", "data"),
        [
            State({"type": _TOGGLE, "index": ALL}, "value"),
            State({"type": _SLIDER, "index": ALL}, "value"),
            State({"type": _DROPDOWN, "index": ALL}, "value"),
        ],
        prevent_initial_call=True,
    )
    def _run_optimisation_and_close(trigger, toggles, sliders, tiers):
        if trigger is None:
            return no_update, no_update, no_update, no_update, no_update
        result = run_optimisation(
            toggles=toggles,
            sliders=sliders,
            tiers=tiers,
        )
        result_text = f"{result['message']} Status: {result['status']}."
        return (
            result_text,
            False,
            None,
            _RESULTS_ACCORDION_ITEM_ID,
            result_text,
        )
