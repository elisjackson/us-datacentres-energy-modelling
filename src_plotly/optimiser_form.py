"""
Optimiser parameters form in three sections: Data Centre, Generation, CO2 settings.
Each section can later have its own columns and components.
When a row's toggle is off, its slider and dropdown are disabled and greyed (values retained).
"""
from dash import dcc, html, Input, Output, State, ALL, no_update
import dash_bootstrap_components as dbc

from src_plotly.optimise import run_optimisation, execute_optimisation

ROW_LABELS = [
    "Data Centre Capacity",
    "Wind",
    "Solar PV",
    "Grid Connection",
    "Gas CCGT",
    "SMR",
    "CO2 price"
]
N_ROWS = len(ROW_LABELS)
ROW_IDS = list(range(N_ROWS))

# Sections: title + row indices. Optional "columns" = list of column ids to show (default: all).
# Column ids: "parameter", "toggle", "slider", "tier"
SECTIONS = [
    {"title": "Data Centre", "row_ids": [0], "columns": ["parameter", "slider"]},
    {"title": "Generation", "row_ids": [1, 2, 3, 4, 5]},
    {"title": "CO2 settings", "row_ids": [6], "columns": ["parameter", "toggle", "tier"], "tier_in_slider_column": True},
]

# Column widths (Bootstrap grid, must sum to 12): Parameter | Toggle | Level | Tier
W_PARAM, W_TOGGLE, W_CAPACITY, W_TIER = 2, 1, 5, 4

# ID type prefixes so callbacks don't clash with other components
_TOGGLE = "optimiser-toggle"
_SLIDER = "optimiser-slider"            # dcc.Slider: single value (all rows currently)
_RANGESLIDER = "optimiser-rangeslider"  # Reserved: dcc.RangeSlider [low, high] for GENERATION_ROW_IDS if re-enabled
_SLIDER_VALUE = "optimiser-slider-value"
_DROPDOWN = "optimiser-dropdown"
_ROW_WRAPPER = "optimiser-row-wrapper"

SINGLE_SLIDER_ROW_IDS = [0, 6]   # Data Centre, CO2 (CO2 slider hidden)
GENERATION_ROW_IDS = [1, 2, 3, 4, 5]  # Reserved for future RangeSlider (low–high) per row

# Optional valid range per row (single sliders only; values outside range are clamped on change).
# e.g. row 0 (Data Centre Capacity): valid range 10-100, so 0-9 snap to 10.
SLIDER_VALID_RANGE: dict[int, dict[str, int]] = {
    0: {"min": 10},  # Data Centre Capacity
}


def _hidden_class(visible_columns: list | None, col_id: str) -> str:
    """Return 'optimiser-col-hidden' when column should be hidden, else ''."""
    if visible_columns is None or col_id in visible_columns:
        return ""
    return "optimiser-col-hidden"


def _slider_col_content(row_id: int):
    """Single dcc.Slider (max value) for all rows. RangeSlider infra kept for future use (see _RANGESLIDER, GENERATION_ROW_IDS)."""
    default = 50
    if row_id == 0 and SLIDER_VALID_RANGE.get(0, {}).get("min") is not None:
        default = max(50, SLIDER_VALID_RANGE[0]["min"])
    return html.Div(
        [
            dcc.Slider(
                id={"type": _SLIDER, "index": row_id},
                min=0,
                max=100,
                step=10,
                value=default,
                marks=None,
                className="row-slider-dcc",
            ),
            html.Span(
                id={"type": _SLIDER_VALUE, "index": row_id},
                className="row-slider-value ms-2 text-muted",
                style={"fontSize": "0.9rem"},
            ),
        ],
        className="d-flex align-items-center row-slider-wrapper",
    )


def _tier_col_content(row_id: int):
    """Tier dropdown."""
    return html.Div(
        dcc.Dropdown(
            id={"type": _DROPDOWN, "index": row_id},
            options=[
                {"label": "Low", "value": "Low"},
                {"label": "Mid", "value": "Mid"},
                {"label": "High", "value": "High"},
            ],
            value="Mid",
            clearable=False,
        ),
        className="optimiser-tier-dropdown-wrapper w-100",
    )


def _make_row(row_id: int, visible_columns: list | None = None, tier_in_slider_column: bool = False):
    """Single parameter row. visible_columns restricts which columns are shown. tier_in_slider_column: show Tier in slider column (CO2 row)."""
    if tier_in_slider_column:
        # Tier appears in slider column; slider stays in DOM in tier column but hidden
        col3_content = _tier_col_content(row_id)
        col4_content = _slider_col_content(row_id)
        col3_hide = ""
        col4_hide = "optimiser-col-hidden"
    else:
        col3_content = _slider_col_content(row_id)
        col4_content = _tier_col_content(row_id)
        col3_hide = _hidden_class(visible_columns, "slider")
        col4_hide = _hidden_class(visible_columns, "tier")

    return dbc.Row(
        [
            dbc.Col(
                html.Span(ROW_LABELS[row_id], className="text-nowrap"),
                width=W_PARAM,
                className=f"d-flex align-items-center {_hidden_class(visible_columns, 'parameter')}".strip(),
            ),
            dbc.Col(
                dbc.Switch(
                    id={"type": _TOGGLE, "index": row_id},
                    label="",
                    value=True,
                    className="mb-0",
                ),
                width=W_TOGGLE,
                className=f"d-flex align-items-center justify-content-center {_hidden_class(visible_columns, 'toggle')}".strip(),
            ),
            dbc.Col(
                col3_content,
                width=W_CAPACITY,
                className=f"d-flex align-items-center justify-content-center {col3_hide}".strip(),
            ),
            dbc.Col(
                col4_content,
                width=W_TIER,
                className=f"d-flex align-items-center justify-content-center {col4_hide}".strip(),
            ),
        ],
        className="mb-3 align-items-center row-controls",
        id={"type": _ROW_WRAPPER, "index": row_id},
    )


def _section_header_row(visible_columns: list | None = None, tier_in_slider_column: bool = False):
    """Column headers. tier_in_slider_column: show 'Tier' in slider column (for CO2 section)."""
    if tier_in_slider_column:
        col3_header = html.Strong("Cost Tier")
        col4_header = html.Strong("Capacity (MW)")
        col3_class = ""
        col4_class = "optimiser-col-hidden"
    else:
        col3_header = html.Strong("Capacity (MW)")
        col4_header = html.Strong("Cost Tier")
        col3_class = _hidden_class(visible_columns, "slider")
        col4_class = _hidden_class(visible_columns, "tier")

    return dbc.Row(
        [
            dbc.Col(html.Strong("Parameter"), width=W_PARAM, className=_hidden_class(visible_columns, "parameter")),
            dbc.Col(html.Strong("Enabled"), width=W_TOGGLE, className=f"text-center {_hidden_class(visible_columns, 'toggle')}".strip()),
            dbc.Col(col3_header, width=W_CAPACITY, className=col3_class),
            dbc.Col(col4_header, width=W_TIER, className=col4_class),
        ],
        className="mb-2 text-muted small",
    )


def _make_section(section: dict, is_first: bool = False):
    """Build one form section: title + header row + parameter rows. Section can set 'columns' and 'tier_in_slider_column'."""
    title = section["title"]
    row_ids = section["row_ids"]
    visible_columns = section.get("columns")
    tier_in_slider_column = section.get("tier_in_slider_column", False)
    title_class = "mb-2 mt-0 fw-semibold" if is_first else "mb-2 mt-4 fw-semibold"
    return html.Div(
        [
            html.H5(title, className=title_class, style={"fontSize": "1.1rem", "borderBottom": "1px solid rgba(255,255,255,0.2)", "paddingBottom": "0.25rem"}),
            _section_header_row(visible_columns, tier_in_slider_column),
            *[_make_row(i, visible_columns, tier_in_slider_column) for i in row_ids],
        ],
        className="optimiser-section",
    )


def optimiser_form_layout():
    """Build the optimiser form: three sections (Data Centre, Generation, CO2) then Optimise button."""
    return html.Div(
        [
            *[_make_section(s, is_first=(i == 0)) for i, s in enumerate(SECTIONS)],
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

    # Clamp slider values to valid range (e.g. Data Centre Capacity min 10)
    @app.callback(
        Output({"type": _SLIDER, "index": ALL}, "value"),
        Input({"type": _SLIDER, "index": ALL}, "value"),
    )
    def _clamp_slider_values(values):
        if values is None:
            return no_update
        result = []
        for i, v in enumerate(values):
            if v is None:
                result.append(50)
                continue
            r = SLIDER_VALID_RANGE.get(i)
            if not r:
                result.append(v)
                continue
            v = int(v)
            if "min" in r and v < r["min"]:
                v = r["min"]
            if "max" in r and v > r["max"]:
                v = r["max"]
            result.append(v)
        return result

    # Show current slider value next to each row
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
        toggles_by_param = dict(zip(ROW_LABELS, toggles))
        sliders_by_param = dict(zip(ROW_LABELS, sliders))
        tiers_by_param = dict(zip(ROW_LABELS, tiers))
        TEST = True
        if TEST:
            result = run_optimisation(
                toggles=toggles_by_param,
                sliders=sliders_by_param,
                tiers=tiers_by_param,
            )
        else:
            result = execute_optimisation(
                toggles=toggles_by_param,
                sliders=sliders_by_param,
                tiers=tiers_by_param,
            )
        result_text = f"{result['message']} Status: {result['status']} Duration: {result['duration_seconds']} seconds."
        return (
            result_text,
            False,
            None,
            _RESULTS_ACCORDION_ITEM_ID,
            result_text,
        )
