import dash_bootstrap_components as dbc
import json
import logging
import pandas as pd
from dash import Input, Output, State, html, dcc, ctx, no_update
from dash.dependencies import ALL, MATCH
from dash.exceptions import PreventUpdate
from pathlib import Path
from typing import List, Literal

# Local imports
try:
    import src.optimiser_api as optimiser_api
except ImportError:
    import optimiser_api as optimiser_api

DIR = Path(__file__).parent
CONFIG_DIR = DIR / "config"

# Fallback map coordinates when no solar/wind location is selected (used by optimiser and modal warning).
DEFAULT_MAP_LOCATION = {"lat": 54.00366, "lon": -2.54786}

class GenerationInput():
    """
    A collapsible card for a generation type.
    """
    # Column widths for cost level rows
    BTN_COL_WIDTH = 4
    INPUT_COL_WIDTH = 8
    
    def __init__(
        self,
        title: str,
        initial_open: bool = False,
        ):
        """
        Args:
            title: the card title (must match a key in GENERATION_CONFIG)
            initial_open: if True, the card body is expanded on load (so radios are visible without clicking).
        """


        self.config = GENERATION_CONFIG[title]
        self.title = title
        self.card_id = f"generation-input-{title.lower().replace(' ', '-')}-card"

        # E.g. ["Onshore", "Offshore"] for Wind
        self.subtypes = list(self.config.keys())

        self.cost_choices_df = self.create_costs_df(cost_type="cost_choices")
        self.cost_assumptions_df = self.create_costs_df(cost_type="cost_assumptions")

        self.cost_choice_params = {}
        self.n_cost_columns = {}
        self.default_subtype = self.subtypes[0] if self.subtypes else None
        self.cost_columns = {}
        self.additional_assumptions_column = {}
        
        # Check if cost_assumptions exist
        self.has_cost_assumptions = not self.cost_assumptions_df.empty
        
        for subtype in self.subtypes:
            self.cost_choice_params[subtype] = [
                c["name"] for c in self.config[subtype].get("cost_choices", [])
            ]
            
            # Calculate number of columns (cost choices + assumptions column if exists)
            num_cost_params = len(self.cost_choice_params[subtype])
            self.n_cost_columns[subtype] = num_cost_params + (1 if self.has_cost_assumptions else 0)

            # Build initial cost columns with default subtype (first choice)
            self.cost_columns[subtype] = self.build_all_cost_columns(subtype)
            
            # build additional assumptions column only if cost_assumptions exist
            if self.has_cost_assumptions:
                self.additional_assumptions_column[subtype] = self.build_additional_assumptions_column(subtype)
                self.cost_columns[subtype].append(self.additional_assumptions_column[subtype])

        if len(self.subtypes) > 1:
            # Create pill buttons for subtypes
            self.subtype_buttons = html.Div(
                [
                    dbc.Button(
                        subtype,
                        id={"type": "subtype-pill", "card": self.card_id, "subtype": subtype},
                        color="primary",
                        outline=(i != 0),  # First button is active (not outlined)
                        active=(i == 0),  # First button is active
                        className="rounded-pill me-2 px-3",
                    )
                    for i, subtype in enumerate(self.subtypes)
                ],
                className="d-flex align-items-center generation-pill-group",
            )
        else:
            self.subtype_buttons = None

        self.card_body = self.create_card_body()
        self.collapsible_card = self.build_collapsible_card(is_open=initial_open)

    def build_additional_assumptions_column(self, subtype: str):
        """Build the additional assumptions column."""
        df = self.cost_assumptions_df.loc[
            self.cost_assumptions_df["subtype"] == subtype
        ]
        dbc_rows = []
        for i, row in df.iterrows():
            dbc_rows.append(
                dbc.Row([
                    dbc.Col(html.Span(row["cost_parameter"]), width=3, className="assumptions-cell"),
                    dbc.Col(html.Span(row["value"]), width=3, className="assumptions-cell assumptions-cell-value"),
                    dbc.Col(html.Span(row["unit"]), width=3, className="assumptions-cell assumptions-cell-unit"),
                    dbc.Col(html.Span(row["source"]), width=3, className="assumptions-cell assumptions-cell-source"),
                ], className="assumptions-data-row")
            )
        return html.Div([
            html.Div("Additional Assumptions", className="assumptions-title"),
            dbc.Row([
                dbc.Col(html.Span("Parameter"), width=3, className="assumptions-col-header"),
                dbc.Col(html.Span("Value"), width=3, className="assumptions-col-header"),
                dbc.Col(html.Span("Unit"), width=3, className="assumptions-col-header"),
                dbc.Col(html.Span("Source"), width=3, className="assumptions-col-header"),
            ], className="assumptions-header-row"),
            *dbc_rows
        ], className="additional-assumptions")

    def create_costs_df(self, cost_type: Literal["cost_choices", "cost_assumptions"]):
        # Flatten to one row per value
        rows = []
        for subtype in self.config:
            # Check if this cost_type exists for this subtype
            if cost_type not in self.config[subtype]:
                continue
            costs = self.config[subtype][cost_type]
            for cost in costs:
                cost_name = cost['name']
                unit = cost['unit']
                for value in cost['values']:
                    rows.append({
                        'subtype': subtype,
                        'cost_parameter': cost_name,
                        'unit': unit,
                        'level': value['name'],
                        'value': value['value'],
                        'source': value['source']
                    })
        if not rows:
            return pd.DataFrame(columns=['subtype', 'cost_parameter', 'unit', 'level', 'value', 'source'])
        return pd.DataFrame(rows)

    def build_all_cost_columns(self, subtype: str, saved_selections=None) -> List[dbc.Row]:
        """Build all cost columns for the given subtype."""
        saved_selections = saved_selections or {}
        return [
            self.build_cost_column(param, subtype, saved_selections.get(param))
            for param in self.cost_choice_params[subtype]
        ]

    def _create_cost_level_row(self, cost_parameter: str, selected_subtype: str, level: str, value=None, is_active=False, input_disabled=False):
        """Helper to create a button + input row for a cost level."""
        if input_disabled:
            input_class = "cost-level-input input-active-disabled" if is_active else "cost-level-input input-inactive-disabled"
        else:
            input_class = "cost-level-input"

        return dbc.Row([
            dbc.Col(
                dbc.Button(
                    level,
                    id={"type": "cost-level-btn", "card": self.card_id, "subtype": selected_subtype, "param": cost_parameter, "level": level},
                    className="cost-level-btn w-100",
                    color="primary",
                    outline=not is_active,
                    active=is_active,
                ),
                width=self.BTN_COL_WIDTH,
            ),
            dbc.Col(
                dbc.Input(
                    type="number",
                    value=value,
                    placeholder="0" if value is not None else "Custom value",
                    disabled=input_disabled,
                    className=input_class,
                    id={"type": "cost-value-input", "card": self.card_id, "subtype": selected_subtype, "param": cost_parameter, "level": level},
                    min=0,
                ),
                width=self.INPUT_COL_WIDTH,
            ),
        ], className="g-0 cost-level-row")
    
    def build_cost_column(self, cost_parameter: str, selected_subtype: str, saved_selection=None):
        """
        Build one cost column, filtered by subtype and cost parameter.

        E.g. CAPEX column containing a title, and a row for each cost level.
        """
        df = self.cost_choices_df.loc[
            (self.cost_choices_df["subtype"] == selected_subtype) &
            (self.cost_choices_df["cost_parameter"] == cost_parameter)
        ]

        if df.empty:
            unit = ""
            for c in self.config[selected_subtype].get("cost_choices", []):
                if c["name"] == cost_parameter:
                    unit = c.get("unit", "")
                    break
            default_level = "Custom"
        else:
            unit = df["unit"].unique()[0]
            # Prefer "Mid" as default selection when available, otherwise fallback to first level.
            default_level = None
            for level in df["level"].tolist():
                if str(level).strip().lower() == "mid":
                    default_level = level
                    break
            if default_level is None:
                default_level = df.iloc[0]["level"]

        available_levels = set(df["level"].tolist())
        available_levels.add("Custom")
        saved_level = (saved_selection or {}).get("level")
        active_level = saved_level if saved_level in available_levels else default_level
        saved_custom_value = (saved_selection or {}).get("custom_value")

        # Build a row for each cost level.
        level_rows = []
        for _, row in df.iterrows():
            row_level = row["level"]
            level_rows.append(
                self._create_cost_level_row(
                    cost_parameter=cost_parameter,
                    selected_subtype=selected_subtype,
                    level=row_level,
                    value=row["value"],
                    is_active=(row_level == active_level),
                    input_disabled=True  # Pre-defined values are read-only
                )
            )
        
        # Append a customizable row (enabled only when Custom button is clicked)
        level_rows.append(
            self._create_cost_level_row(
                cost_parameter=cost_parameter,
                selected_subtype=selected_subtype,
                level="Custom",
                value=saved_custom_value,
                is_active=(active_level == "Custom"),
                input_disabled=(active_level != "Custom")  # Enable when Custom is selected
            )
        )
        
        # Build source attribution text
        if df.empty:
            source_text = ""
        else:
            sources = df[["level", "source"]].drop_duplicates()
            if sources["source"].nunique() == 1:
                source_text = f"Source: {sources['source'].iloc[0]}"
            else:
                parts = [f"{row['level']}: {row['source']}" for _, row in sources.iterrows()]
                source_text = f"Source: {'; '.join(parts)}"

        # Label row on top, level rows below, source at the bottom
        return html.Div([
            html.Label(f"Select {cost_parameter} ({unit})", className="cost-column-header text-label-blue"),
            *level_rows,
            html.P(source_text, className="text-muted-small mt-2"),
        ])

    def build_collapsible_card(self, is_open: bool = False):
        """
        Reusable card with always-visible header and click-to-toggle body.
        """
        icon_class = "bi bi-chevron-up" if is_open else "bi bi-chevron-down"
        return dbc.Card(
            [
                dbc.CardHeader(
                    html.Button(
                        [
                            html.Span(self.title),
                            html.I(
                                id={"type": "collapse-card-icon", "index": self.card_id},
                                className=icon_class,
                                **{"aria-hidden": "true"},
                            ),
                        ],
                        id={"type": "collapse-card-toggle", "index": self.card_id},
                        type="button",
                        **{"aria-expanded": str(is_open).lower()},
                        n_clicks=0,
                        className="w-100 d-flex justify-content-between align-items-center border-0 bg-transparent text-start p-0",
                    )
                ),
                dbc.Collapse(
                    self.card_body,
                    id={"type": "collapse-card-body", "index": self.card_id},
                    is_open=is_open,
                ),
            ],
            className="mb-3 generation-input-card",
        )

    def create_card_body(self):
        children = []
        
        # Subtype pill buttons row
        if self.subtype_buttons is not None:
            children.append(
                dbc.Row([
                    dbc.Col(html.Label("Select base cost assumptions:", className="text-label-blue"), width="auto"),
                    dbc.Col(self.subtype_buttons, width="auto")
                ], className="align-items-center mb-3")
                )
        
        # Add cost columns in a container that can be updated dynamically
        children.append(
            html.Div(
                dbc.Row([
                    dbc.Col(
                        c,
                        className=(
                            "assumptions-flex-col"
                            if "additional-assumptions" in (getattr(c, "className", "") or "")
                            else "cost-parameter-col"
                        ),
                    )
                    for c in self.cost_columns[self.default_subtype]
                ], className="mb-3 cost-columns-row align-items-center"),
                id={"type": "cost-columns-container", "card": self.card_id}
            )
        )
        
        return dbc.CardBody(children, className="py-3")


# read generation form from config/generation_form.json
with open(CONFIG_DIR / "technology_costs.json", "r") as f:
    GENERATION_CONFIG = json.load(f)


# grid_electricity_card = GenerationInput("Grid Electricity", initial_open=True)
wind_card = GenerationInput("Wind")
solar_card = GenerationInput("Solar", initial_open=True)
gas_card = GenerationInput("Gas")
smr_card = GenerationInput("SMR")
battery_storage_card = GenerationInput("Battery storage")
co2_card = GenerationInput("Carbon price")
generation_cards = [solar_card, wind_card, gas_card, smr_card]
GENERATION_CARDS_ACTIVE_BY_DEFAULT = (solar_card, wind_card, gas_card)
generation_card_ids = [card.card_id for card in generation_cards]
# Store cards by ID for callback access (include battery storage and CO2 for callbacks)
generation_cards_dict = {card.card_id: card for card in generation_cards}
generation_cards_dict[battery_storage_card.card_id] = battery_storage_card
generation_cards_dict[co2_card.card_id] = co2_card


def form_layout():
    """Return the concept form layout for use in main app."""
    return html.Div(
        [
            # Data center parameters section
            html.H5("Data center parameters", className="mb-3"),
            # Data centre capacity row
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Data centre capacity (MW)", className="text-label-blue"), width="auto", className="d-flex align-items-center"),
                    dbc.Col(
                        dcc.Slider(
                            id="data-centre-capacity-slider",
                            min=10,
                            max=120,
                            step=1,
                            value=10,
                        ),
                        width=True,
                    ),
                ],
                className="mb-4 pb-3 align-items-center",
            ),
            # Energy system parameters section
            html.H5("Energy system parameters", className="mb-3"),
            # Selection controls row
            dbc.Row(
                [
                    # Generation selection column
                    dbc.Col(
                        [
                            dbc.Label("Select generation / energy sources", className="mb-2 text-center text-label-blue"),
                            html.Div(
                                [
                                    dbc.Button(
                                        card.title,
                                        id={"type": "generation-pill", "index": card.card_id},
                                        color="primary",
                                        outline=(card not in GENERATION_CARDS_ACTIVE_BY_DEFAULT),
                                        active=(card in GENERATION_CARDS_ACTIVE_BY_DEFAULT),
                                        className="rounded-pill me-2 px-3",
                                    )
                                    for card in generation_cards
                                ],
                                className="d-flex align-items-center justify-content-center generation-pill-group",
                            ),
                        ],
                        width="auto",
                    ),
                    # Battery storage column
                    dbc.Col(
                        [
                            dbc.Label("Battery storage", className="mb-2 text-center text-label-blue"),
                            html.Div(
                                dbc.Switch(
                                    id={"type": "storage-toggle", "index": "storage-toggle"},
                                    value=False
                                ),
                                className="d-flex align-items-center justify-content-center",
                                style={"minHeight": "38px"}  # Match button height
                            ),
                        ],
                        width="auto",
                    ),
                    # Carbon price column
                    dbc.Col(
                        [
                            dbc.Label("Include Carbon price", className="mb-2 text-center text-label-blue"),
                            html.Div(
                                dbc.Switch(
                                    id={"type": "co2-toggle", "index": "co2-toggle"},
                                    value=False
                                ),
                                className="d-flex align-items-center justify-content-center",
                                style={"minHeight": "38px"}  # Match button height
                            ),
                        ],
                        width="auto",
                    ),
                ],
                className="mb-3 align-items-start justify-content-between justify-content-lg-start gap-lg-5",
            ),
            dbc.Row(
                [
                    html.Div(
                        dbc.Col(
                            card.collapsible_card,
                            width=12,
                        ),
                        id={"type": "generation-card-col", "index": card.card_id},
                        className="card-container-animated " + ("card-visible" if card in GENERATION_CARDS_ACTIVE_BY_DEFAULT else "card-hidden"),
                    )
                    for card in generation_cards
                ],
            ),
            # Battery storage card (shown/hidden by callback)
            html.Div(
                dbc.Row([
                    dbc.Col(
                        battery_storage_card.collapsible_card,
                        width=12,
                    )
                ]),
                id="battery-storage-card-container",
                className="card-container-animated card-hidden"  # Hidden by default with animation
            ),
            # CO2 card (shown/hidden by callback)
            html.Div(
                dbc.Row([
                    dbc.Col(
                        co2_card.collapsible_card,
                        width=12,
                    )
                ]),
                id="co2-card-container",
                className="card-container-animated card-hidden"  # Hidden by default with animation
            ),
            # Warning box above Optimise button (shown only when button is inactive)
            html.Div(id="optimise-button-warning", children=[], style={"display": "none"}),
            # Optimise button at the bottom
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Optimise",
                            id="optimise-button",
                            color="primary",
                            className="w-100 mt-2",
                            disabled=False,  # Will be controlled by callback
                        ),
                        width=12,
                    )
                ],
            ),
            # Stores for optimisation workflow
            dcc.Store(id="optimiser-parameters-store"),
            dcc.Store(id="optimiser-trigger-run"),
            dcc.Store(id="optimiser-job-store"),
            dcc.Store(id="optimiser-results-fetched-job-id"),
            dcc.Store(id="cost-selections-store", data={}),
            dcc.Interval(id="optimiser-poll-interval", interval=2000, n_intervals=0, disabled=True),
            # Loading modal (body message is updated when infeasible)
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Baking the cheapest solution...")),
                    dbc.ModalBody(
                        [
                            html.Div(id="optimiser-loading-modal-warning", className="mb-2"),
                            html.Div(
                                id="optimiser-loading-modal-message",
                                children=[
                                    dbc.Spinner(color="primary", size="sm", spinner_class_name="me-2"),
                                    " Queued for optimisation",
                                ],
                                className="d-flex align-items-center",
                            ),
                        ],
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id="optimiser-loading-modal-close",
                            color="secondary",
                            outline=True,
                        )
                    ),
                ],
                id="optimiser-loading-modal",
                className="app-modal",
                is_open=False,
                centered=True,
                backdrop="static",
                keyboard=False,
            ),
        ],
        style={
            "backgroundColor": "rgb(5, 13, 24)",
            "borderRadius": "8px",
            "padding": "1rem",
            "border": "1px solid rgb(111, 111, 111)"
        },
    )


def register_callbacks(app):
    """Register all callbacks for the concept form."""
    
    @app.callback(
        Output({"type": "collapse-card-body", "index": MATCH}, "is_open"),
        Output({"type": "collapse-card-icon", "index": MATCH}, "className"),
        Output({"type": "collapse-card-toggle", "index": MATCH}, "aria-expanded"),
        Input({"type": "collapse-card-toggle", "index": MATCH}, "n_clicks"),
        State({"type": "collapse-card-body", "index": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_collapsible_card(_n_clicks, is_open):
        # State can be None on first load; treat as closed so first click opens
        is_open = is_open if is_open is not None else False
        new_is_open = not is_open
        icon_class = "bi bi-chevron-up" if new_is_open else "bi bi-chevron-down"
        return new_is_open, icon_class, str(new_is_open).lower()

    
    @app.callback(
        Output({"type": "subtype-pill", "card": MATCH, "subtype": ALL}, "active"),
        Output({"type": "subtype-pill", "card": MATCH, "subtype": ALL}, "outline"),
        Output({"type": "cost-columns-container", "card": MATCH}, "children"),
        Input({"type": "subtype-pill", "card": MATCH, "subtype": ALL}, "n_clicks"),
        State({"type": "subtype-pill", "card": MATCH, "subtype": ALL}, "id"),
        State({"type": "subtype-pill", "card": MATCH, "subtype": ALL}, "active"),
        State({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "active"),
        State({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "value"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
        State("cost-selections-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_subtype_pills_and_update_columns(
        n_clicks_list,
        pill_ids,
        current_pill_active,
        cost_btn_active,
        cost_btn_ids,
        cost_input_values,
        cost_input_ids,
        stored_cost_selections,
    ):
        """Toggle subtype pills and update cost columns when a subtype is selected."""
        if not ctx.triggered_id:
            return no_update, no_update, no_update
        
        # Find which pill was clicked
        clicked_pill = ctx.triggered_id
        card_id = clicked_pill["card"]
        selected_subtype = clicked_pill["subtype"]
        
        # Get the card instance
        card = generation_cards_dict.get(card_id)
        if not card:
            return no_update, no_update, no_update

        stored_cost_selections = stored_cost_selections or {}

        # Persist current subtype selections before switching away.
        current_subtype = None
        for is_active, pill in zip(current_pill_active, pill_ids):
            if is_active:
                current_subtype = pill["subtype"]
                break

        if current_subtype:
            card_store = stored_cost_selections.setdefault(card_id, {})
            current_subtype_store = card_store.setdefault(current_subtype, {})

            active_by_param = {}
            for is_active, btn_id in zip(cost_btn_active, cost_btn_ids):
                if (
                    is_active
                    and btn_id["card"] == card_id
                    and btn_id["subtype"] == current_subtype
                ):
                    active_by_param[btn_id["param"]] = btn_id["level"]

            custom_values_by_param = {}
            for value, inp_id in zip(cost_input_values, cost_input_ids):
                if (
                    inp_id["card"] == card_id
                    and inp_id["subtype"] == current_subtype
                    and inp_id["level"] == "Custom"
                ):
                    custom_values_by_param[inp_id["param"]] = value

            for param, level in active_by_param.items():
                current_subtype_store[param] = {
                    "level": level,
                    "custom_value": custom_values_by_param.get(param),
                }
        
        # Update active states: only clicked pill is active
        active_states = [pill["subtype"] == selected_subtype for pill in pill_ids]
        outline_states = [not active for active in active_states]
        
        # Rebuild cost columns for the new subtype
        selected_store = (
            stored_cost_selections.get(card_id, {}).get(selected_subtype, {})
        )
        cost_columns = card.build_all_cost_columns(selected_subtype, selected_store)
        
        # Add the additional assumptions column for this subtype (if it exists)
        if card.has_cost_assumptions:
            cost_columns.append(card.additional_assumptions_column[selected_subtype])
        
        # Return the updated states and columns
        columns_row = dbc.Row([
            dbc.Col(
                c,
                className=(
                    "assumptions-flex-col"
                    if "additional-assumptions" in (getattr(c, "className", "") or "")
                    else "cost-parameter-col"
                ),
            )
            for c in cost_columns
        ], className="mb-3 cost-columns-row align-items-center")
        
        return active_states, outline_states, columns_row
    
    
    @app.callback(
        Output({"type": "generation-pill", "index": ALL}, "active"),
        Output({"type": "generation-pill", "index": ALL}, "outline"),
        Output({"type": "generation-card-col", "index": ALL}, "className"),
        Input({"type": "generation-pill", "index": ALL}, "n_clicks"),
        State({"type": "generation-pill", "index": ALL}, "active"),
        prevent_initial_call=True,
    )
    def toggle_generation_cards(_pill_clicks, pill_active_states):
        active_states = [bool(state) for state in pill_active_states]
        triggered_pill = ctx.triggered_id

        if isinstance(triggered_pill, dict):
            triggered_card_id = triggered_pill.get("index")
            if triggered_card_id in generation_card_ids:
                clicked_index = generation_card_ids.index(triggered_card_id)
                active_states[clicked_index] = not active_states[clicked_index]

        card_classes = [
            "card-container-animated card-visible" if is_active else "card-container-animated card-hidden"
            for is_active in active_states
        ]
        pill_outlines = [not is_active for is_active in active_states]

        return active_states, pill_outlines, card_classes
    
    
    @app.callback(
        Output({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "active"),
        Output({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "outline"),
        Output({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "disabled"),
        Output({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "className"),
        Output("cost-selections-store", "data"),
        Input({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "n_clicks"),
        State({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "active"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "value"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
        State("cost-selections-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_cost_level_buttons(
        n_clicks_list,
        current_active_states,
        cost_input_values,
        cost_input_ids,
        stored_cost_selections,
    ):
        """Make only the clicked button active within its cost parameter group (like radio buttons).
        Also enable Custom input only when Custom button is selected."""
        from dash import callback_context
        
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate

        # Ignore mount/re-render events where no button was actually clicked.
        triggered_prop = callback_context.triggered[0] if callback_context.triggered else None
        triggered_clicks = triggered_prop.get("value") if triggered_prop else None
        if not triggered_clicks:
            raise PreventUpdate
        
        # Get which button was clicked
        clicked_card = triggered["card"]
        clicked_subtype = triggered["subtype"]
        clicked_param = triggered["param"]
        clicked_level = triggered["level"]
        
        # Get all button IDs from the outputs (outputs_list is a list of dicts, one per Output)
        outputs_list = callback_context.outputs_list
        # First output is "active", get the list of component specs
        button_ids = [output["id"] for output in outputs_list[0]]
        input_ids = [output["id"] for output in outputs_list[2]]  # Third output is inputs
        
        # Build active/outline states for all buttons
        active_states = []
        outline_states = []
        
        for i, btn_id in enumerate(button_ids):
            # Check if this button is in the same (card, subtype, param) group as the clicked button
            same_group = (
                btn_id["card"] == clicked_card
                and btn_id["subtype"] == clicked_subtype
                and btn_id["param"] == clicked_param
            )
            
            if same_group:
                # Within the clicked group: only the clicked button is active
                is_clicked_button = (btn_id["level"] == clicked_level)
                active_states.append(is_clicked_button)
                outline_states.append(not is_clicked_button)
            else:
                # Different group: preserve current state
                active_states.append(current_active_states[i])
                outline_states.append(not current_active_states[i])
        
        # Build disabled states and class names for all inputs
        input_disabled_states = []
        input_classes = []

        for inp_id in input_ids:
            # Custom inputs are enabled only if their corresponding Custom button is active
            if inp_id["level"] == "Custom":
                custom_btn_active = False
                for i, btn_id in enumerate(button_ids):
                    if (btn_id["card"] == inp_id["card"] and
                            btn_id["subtype"] == inp_id["subtype"] and
                            btn_id["param"] == inp_id["param"] and
                            btn_id["level"] == "Custom"):
                        custom_btn_active = active_states[i]
                        break
                is_disabled = not custom_btn_active
                input_disabled_states.append(is_disabled)
                input_classes.append("cost-level-input input-inactive-disabled" if is_disabled else "cost-level-input")
            else:
                # Non-custom inputs are always disabled; highlight if their button is active
                btn_is_active = False
                for i, btn_id in enumerate(button_ids):
                    if (btn_id["card"] == inp_id["card"] and
                            btn_id["subtype"] == inp_id["subtype"] and
                            btn_id["param"] == inp_id["param"] and
                            btn_id["level"] == inp_id["level"]):
                        btn_is_active = active_states[i]
                        break
                input_disabled_states.append(True)
                input_classes.append("cost-level-input input-active-disabled" if btn_is_active else "cost-level-input input-inactive-disabled")

        stored_cost_selections = stored_cost_selections or {}
        card_store = stored_cost_selections.setdefault(clicked_card, {})
        subtype_store = card_store.setdefault(clicked_subtype, {})

        custom_value = None
        for value, inp_id in zip(cost_input_values, cost_input_ids):
            if (
                inp_id["card"] == clicked_card
                and inp_id["subtype"] == clicked_subtype
                and inp_id["param"] == clicked_param
                and inp_id["level"] == "Custom"
            ):
                custom_value = value
                break

        subtype_store[clicked_param] = {
            "level": clicked_level,
            "custom_value": custom_value,
        }

        return active_states, outline_states, input_disabled_states, input_classes, stored_cost_selections
    
    
    @app.callback(
        Output("battery-storage-card-container", "className"),
        Input({"type": "storage-toggle", "index": "storage-toggle"}, "value"),
    )
    def toggle_battery_storage_card(storage_enabled):
        """Show or hide the Battery Storage card based on the storage toggle."""
        if storage_enabled:
            return "card-container-animated card-visible"
        else:
            return "card-container-animated card-hidden"
    
    
    @app.callback(
        Output("co2-card-container", "className"),
        Input({"type": "co2-toggle", "index": "co2-toggle"}, "value"),
    )
    def toggle_co2_card(co2_enabled):
        """Show or hide the CO2 card based on the carbon price toggle."""
        if co2_enabled:
            return "card-container-animated card-visible"
        else:
            return "card-container-animated card-hidden"
    
    
    @app.callback(
        Output("optimise-button", "disabled"),
        Output("optimise-button-warning", "children"),
        Output("optimise-button-warning", "style"),
        Input({"type": "generation-pill", "index": ALL}, "active"),
        Input({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "active"),
        Input({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "value"),
        State({"type": "generation-pill", "index": ALL}, "id"),
        State({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
    )
    def update_optimise_button(
        pill_active_states,
        cost_btn_active,
        cost_input_values,
        pill_ids,
        cost_btn_ids,
        cost_input_ids,
    ):
        """Disable Optimise if no generation selected, or if any active card has Custom with value 0 or empty. Show warning when disabled."""
        # Active generation card ids (solar, wind, gas, SMR only)
        if pill_ids is None:
            active_gen_card_ids = set()
        else:
            active_gen_card_ids = {
                pid["index"]
                for active, pid in zip(pill_active_states or [], pill_ids)
                if active and pid.get("index") in generation_card_ids
            }
        has_active = len(active_gen_card_ids) > 0
        if not has_active:
            return True, dbc.Alert(
                "⚠️ Select at least one generation source to run the optimiser.",
                style={
                    "backgroundColor": "#09131f",
                    "color": "#ecf0f1",
                    "border": "solid",
                    "borderColor": "#ffea5dc2",
                    "borderWidth": "thin",
                },
            ), {}
        # Check for Custom selected with value 0 or empty in any active generation card
        if cost_btn_ids is None or cost_input_ids is None:
            return False, [], {"display": "none"}
        for btn_active, btn_id in zip(cost_btn_active or [], cost_btn_ids):
            if btn_id.get("card") not in active_gen_card_ids:
                continue
            if btn_id.get("level") != "Custom" or not btn_active:
                continue
            # Find matching input value
            for j, inp_id in enumerate(cost_input_ids):
                if (
                    inp_id.get("card") == btn_id.get("card")
                    and inp_id.get("subtype") == btn_id.get("subtype")
                    and inp_id.get("param") == btn_id.get("param")
                    and inp_id.get("level") == btn_id.get("level")
                ):
                    val = cost_input_values[j] if cost_input_values and j < len(cost_input_values) else None
                    if val is None or val == "" or (isinstance(val, (int, float)) and val == 0):
                        return True, dbc.Alert(
                "⚠️ Enter a value greater than 0 for all Custom cost options in the active generation cards.",
                style={
                    "backgroundColor": "#09131f",
                    "color": "#ecf0f1",
                    "border": "solid",
                    "borderColor": "#ffea5dc2",
                    "borderWidth": "thin",
                },
            ), {}
                    break
        return False, [], {"display": "none"}
    
    
    # Default content for loading modal (reset when modal opens)
    _loading_modal_default = [
        dbc.Spinner(color="primary", size="sm", spinner_class_name="me-2"),
        " Queued for optimisation",
    ]

    # Optimise button: open loading modal and trigger run
    @app.callback(
        Output("optimiser-loading-modal", "is_open"),
        Output("optimiser-trigger-run", "data"),
        Output("optimiser-loading-modal-warning", "children"),
        Output("optimiser-loading-modal-message", "children"),
        Input("optimise-button", "n_clicks"),
        State("pv-latlon-store", "data"),
        State("wind-latlon-store", "data"),
        State({"type": "generation-pill", "index": ALL}, "active"),
        State({"type": "generation-pill", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def open_loading_modal(n_clicks, pv_latlon, wind_latlon, gen_pill_active, gen_pill_ids):
        """Open loading modal when Optimise button is clicked. Show warning if default location is used for Solar/Wind."""
        valid_pv = (
            isinstance(pv_latlon, dict)
            and "lat" in pv_latlon
            and "lon" in pv_latlon
        )
        valid_wind = (
            isinstance(wind_latlon, dict)
            and "lat" in wind_latlon
            and "lon" in wind_latlon
        )
        solar_active = any(
            active and pid["index"] == solar_card.card_id
            for active, pid in zip(gen_pill_active or [], gen_pill_ids or [])
        )
        wind_active = any(
            active and pid["index"] == wind_card.card_id
            for active, pid in zip(gen_pill_active or [], gen_pill_ids or [])
        )
        use_default_solar = solar_active and not valid_pv
        use_default_wind = wind_active and not valid_wind

        default_loc_str = f"{DEFAULT_MAP_LOCATION['lat']}°N, {DEFAULT_MAP_LOCATION['lon']}°W"
        # warning_text = "No map location"
        if use_default_solar and use_default_wind:
            warning_text = f"No Solar and Wind map locations selected."
        elif use_default_solar:
            warning_text = f"No Solar map location selected."
        elif use_default_wind:
            warning_text = f"No Wind map location selected."
        else:
            warning_text = None

        if warning_text:
            warning_text += f" Using default location ({default_loc_str})."
            warning_children = html.Div(warning_text, className="text-warning mb-2")
            message_children = html.Div(
                _loading_modal_default,
                className="d-flex align-items-center",
            )
        else:
            warning_children = []
            message_children = _loading_modal_default

        return True, n_clicks, warning_children, message_children
    
    
    def _running_message(progress_text):
        return html.Div(
            [
                dbc.Spinner(color="primary", size="sm", spinner_class_name="me-2"),
                f" {progress_text}",
            ],
            className="d-flex align-items-center",
        )

    def _infeasible_message():
        return html.Div(
            (
                "Gosh, I'm infeasible :'(\n"
                "Probably because I can't meet the load all year.\n"
                "I'd appreciate some baseload or backup generation... (add Gas or SMR)."
            ),
            style={"whiteSpace": "pre-line"},
        )

    @app.callback(
        Output("optimiser-loading-modal", "is_open", allow_duplicate=True),
        Input("optimiser-loading-modal-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_loading_modal(_n_clicks):
        return False

    @app.callback(
        Output("optimiser-parameters-store", "data"),
        Output("optimiser-job-store", "data"),
        Output("optimiser-poll-interval", "disabled"),
        Output("optimiser-poll-interval", "n_intervals"),
        Output("optimiser-loading-modal", "is_open", allow_duplicate=True),
        Output("main-accordion", "active_item"),
        Output("optimiser-results-data", "data"),
        Output("optimiser-loading-modal-message", "children", allow_duplicate=True),
        Output("optimiser-results-fetched-job-id", "data"),
        Input("optimiser-trigger-run", "data"),
        State("data-centre-capacity-slider", "value"),
        State({"type": "generation-pill", "index": ALL}, "active"),
        State({"type": "generation-pill", "index": ALL}, "id"),
        State({"type": "subtype-pill", "card": ALL, "subtype": ALL}, "active"),
        State({"type": "subtype-pill", "card": ALL, "subtype": ALL}, "id"),
        State({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "active"),
        State({"type": "cost-level-btn", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "value"),
        State({"type": "cost-value-input", "card": ALL, "subtype": ALL, "param": ALL, "level": ALL}, "id"),
        State({"type": "storage-toggle", "index": "storage-toggle"}, "value"),
        State({"type": "co2-toggle", "index": "co2-toggle"}, "value"),
        State("pv-latlon-store", "data"),
        State("wind-latlon-store", "data"),
        State("hub-heights", "data"),
        State("wind-onshore-store", "data"),
        prevent_initial_call=True,
    )
    def collect_optimiser_parameters(
        trigger,
        dc_capacity,
        gen_pill_active,
        gen_pill_ids,
        subtype_pill_active,
        subtype_pill_ids,
        cost_btn_active,
        cost_btn_ids,
        cost_input_values,
        cost_input_ids,
        storage_enabled,
        co2_enabled,
        pv_latlon,
        wind_latlon,
        hub_heights,
        wind_onshore,
    ):
        """Collect all form parameters, submit the optimiser job, and start polling."""

        if trigger is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        # Fallback map coordinates so optimiser can run without selected map points.
        default_pv_location = dict(DEFAULT_MAP_LOCATION)
        default_wind_location = {**DEFAULT_MAP_LOCATION, "onshore": True}

        pv_location = (
            pv_latlon
            if isinstance(pv_latlon, dict)
            and "lat" in pv_latlon
            and "lon" in pv_latlon
            else default_pv_location
        )
        wind_location = (
            {
                "lat": wind_latlon["lat"],
                "lon": wind_latlon["lon"],
                "onshore": bool(wind_latlon.get("onshore", True)),
            }
            if isinstance(wind_latlon, dict)
            and "lat" in wind_latlon
            and "lon" in wind_latlon
            else default_wind_location
        )
        
        # Build the parameters dictionary
        parameters = {
            'data_centre_capacity': dc_capacity,
            'generation': {},
            'battery_storage': {'enabled': storage_enabled},
            'co2': {'enabled': co2_enabled},
        }
        
        # Identify which generation types are enabled
        active_gen_cards = []
        for i, (active, pill_id) in enumerate(zip(gen_pill_active, gen_pill_ids)):
            if active:
                card_id = pill_id["index"]
                # Find the card instance
                card = generation_cards_dict.get(card_id)
                if card:
                    active_gen_cards.append(card)
        
        # For each active generation card, extract its parameters
        for card in active_gen_cards:
            gen_type = card.title
            card_id = card.card_id
            
            # Find selected subtype (if multiple subtypes exist)
            selected_subtype = None
            if len(card.subtypes) > 1:
                # Find which subtype pill is active for this card
                for active, pill_id in zip(subtype_pill_active, subtype_pill_ids):
                    if active and pill_id["card"] == card_id:
                        selected_subtype = pill_id["subtype"]
                        break
            else:
                # Single subtype, use default
                selected_subtype = card.subtypes[0]
            
            if not selected_subtype:
                continue
            
            # Extract costs for this card
            costs = {}
            # Find which cost level button is active for each cost parameter
            for i, (active, btn_id) in enumerate(zip(cost_btn_active, cost_btn_ids)):
                if (
                    btn_id["card"] == card_id
                    and btn_id["subtype"] == selected_subtype
                    and active
                ):
                    param = btn_id["param"]
                    level = btn_id["level"]
                    
                    # Get the corresponding input value
                    for j, inp_id in enumerate(cost_input_ids):
                        if (inp_id["card"] == card_id and 
                            inp_id["subtype"] == selected_subtype and
                            inp_id["param"] == param and 
                            inp_id["level"] == level):
                            value = cost_input_values[j]
                            
                            # Get unit from DataFrame
                            unit_df = card.cost_choices_df[
                                (card.cost_choices_df["subtype"] == selected_subtype) &
                                (card.cost_choices_df["cost_parameter"] == param)
                            ]
                            unit = unit_df["unit"].unique()[0] if not unit_df.empty else ""
                            
                            costs[param] = {
                                'value': value,
                                'unit': unit
                            }
                            break
            
            # Extract additional assumptions for this subtype and add to costs
            if card.has_cost_assumptions:
                assumptions_df = card.cost_assumptions_df[
                    card.cost_assumptions_df["subtype"] == selected_subtype
                ]
                for _, row in assumptions_df.iterrows():
                    costs[row["cost_parameter"]] = {
                        'value': row["value"],
                        'unit': row["unit"],
                    }
            
            # Build entry for this generation type
            parameters['generation'][gen_type] = {
                'enabled': True,
                'subtype': selected_subtype,
                'costs': costs
            }

            # if type is solar or wind, add additional information
            if gen_type == "Solar":
                parameters['generation'][gen_type]['location'] = pv_location
            elif gen_type == "Wind":
                parameters['generation'][gen_type]['location'] = wind_location
                onshore = wind_onshore if wind_onshore is not None else True
                hub_heights_dict = hub_heights or {}
                hub_height = hub_heights_dict.get("onshore", 150) if onshore else hub_heights_dict.get("offshore", 200)
                parameters['generation'][gen_type]['costs']['Hub height'] = {
                    'value': hub_height,
                    'unit': 'm'
                    }
        
        # Extract Battery Storage parameters if enabled
        if storage_enabled:
            card = battery_storage_card
            card_id = card.card_id
            selected_subtype = card.subtypes[0]  # Battery Storage has single subtype
            
            # Extract costs
            costs = {}
            for i, (active, btn_id) in enumerate(zip(cost_btn_active, cost_btn_ids)):
                if (
                    btn_id["card"] == card_id
                    and btn_id["subtype"] == selected_subtype
                    and active
                ):
                    param = btn_id["param"]
                    level = btn_id["level"]
                    
                    for j, inp_id in enumerate(cost_input_ids):
                        if (inp_id["card"] == card_id and 
                            inp_id["subtype"] == selected_subtype and
                            inp_id["param"] == param and 
                            inp_id["level"] == level):
                            value = cost_input_values[j]
                            
                            unit_df = card.cost_choices_df[
                                (card.cost_choices_df["subtype"] == selected_subtype) &
                                (card.cost_choices_df["cost_parameter"] == param)
                            ]
                            unit = unit_df["unit"].unique()[0] if not unit_df.empty else ""
                            
                            costs[param] = {
                                'value': value,
                                'unit': unit
                            }
                            break
            
            # Extract assumptions and add to costs
            if card.has_cost_assumptions:
                assumptions_df = card.cost_assumptions_df[
                    card.cost_assumptions_df["subtype"] == selected_subtype
                ]
                for _, row in assumptions_df.iterrows():
                    costs[row["cost_parameter"]] = {
                        'value': row["value"],
                        'unit': row["unit"],
                        
                    }
            
            parameters['battery_storage'] = {
                'enabled': True,
                'subtype': selected_subtype,
                'costs': costs
            }
        
        # Extract CO2 parameters if enabled
        if co2_enabled:
            card = co2_card
            card_id = card.card_id
            selected_subtype = card.subtypes[0]  # CO2 has single subtype
            
            # Extract costs
            costs = {}
            for i, (active, btn_id) in enumerate(zip(cost_btn_active, cost_btn_ids)):
                if (
                    btn_id["card"] == card_id
                    and btn_id["subtype"] == selected_subtype
                    and active
                ):
                    param = btn_id["param"]
                    level = btn_id["level"]
                    
                    for j, inp_id in enumerate(cost_input_ids):
                        if (inp_id["card"] == card_id and 
                            inp_id["subtype"] == selected_subtype and
                            inp_id["param"] == param and 
                            inp_id["level"] == level):
                            value = cost_input_values[j]
                            
                            unit_df = card.cost_choices_df[
                                (card.cost_choices_df["subtype"] == selected_subtype) &
                                (card.cost_choices_df["cost_parameter"] == param)
                            ]
                            unit = unit_df["unit"].unique()[0] if not unit_df.empty else ""
                            
                            costs[param] = {
                                'level': level,
                                'value': value,
                                'unit': unit
                            }
                            break
            
            # CO2 has no assumptions
            parameters['co2'] = {
                'enabled': True,
                'subtype': selected_subtype,
                'costs': costs
            }

        try:
            job = optimiser_api.submit_job(parameters)
        except Exception as e:
            logging.exception("Optimiser submission failed")
            optimiser_results = {
                "status": "error",
                "message": str(e),
            }
            return parameters, None, True, 0, False, "accordion-results", optimiser_results, no_update, no_update

        queue_message = _running_message(job.get("progress", "Queued for optimisation"))
        return parameters, job, False, 0, no_update, no_update, no_update, queue_message, None

    @app.callback(
        Output("optimiser-job-store", "data", allow_duplicate=True),
        Output("optimiser-poll-interval", "disabled", allow_duplicate=True),
        Output("optimiser-loading-modal", "is_open", allow_duplicate=True),
        Output("main-accordion", "active_item", allow_duplicate=True),
        Output("optimiser-results-data", "data", allow_duplicate=True),
        Output("optimiser-loading-modal-message", "children", allow_duplicate=True),
        Output("optimiser-results-fetched-job-id", "data", allow_duplicate=True),
        Input("optimiser-poll-interval", "n_intervals"),
        State("optimiser-job-store", "data"),
        State("optimiser-results-fetched-job-id", "data"),
        prevent_initial_call=True,
    )
    def poll_optimiser_job(_n_intervals, job_data, results_fetched_job_id):
        if not job_data or not job_data.get("job_id"):
            raise PreventUpdate

        job_id = job_data["job_id"]

        try:
            status = optimiser_api.get_job_status(job_id)
        except Exception as e:
            logging.exception("Optimiser status check failed")
            optimiser_results = {
                "status": "error",
                "message": str(e),
            }
            return None, True, False, "accordion-results", optimiser_results, no_update, no_update

        job_status = status.get("status")
        progress = status.get("progress", "Running optimisation…")

        if job_status in {"queued", "running"}:
            return status, False, no_update, no_update, no_update, _running_message(progress), no_update

        if job_status == "infeasible":
            return None, True, True, no_update, no_update, _infeasible_message(), no_update

        if job_status in {"done", "error"}:
            if results_fetched_job_id == job_id:
                return None, True, False, "accordion-results", no_update, _loading_modal_default, no_update
            # Stop polling immediately; do not fetch here. A separate callback fetches the result
            # so that duplicate interval ticks (before disabled=True is applied) don't cause
            # duplicate GET optimise-status / GET optimise-result calls.
            return None, True, False, "accordion-results", no_update, no_update, job_id

        optimiser_results = {
            "status": "error",
            "message": f"Unexpected optimiser job status: {job_status}",
        }
        return None, True, False, "accordion-results", optimiser_results, _loading_modal_default, no_update

    @app.callback(
        Output("optimiser-results-data", "data", allow_duplicate=True),
        Output("optimiser-loading-modal", "is_open", allow_duplicate=True),
        Output("optimiser-loading-modal-message", "children", allow_duplicate=True),
        Input("optimiser-results-fetched-job-id", "data"),
        State("optimiser-results-data", "data"),
        prevent_initial_call=True,
    )
    def fetch_optimiser_result_when_ready(job_id, existing_results):
        """Fetch job result when poll callback has set results_fetched_job_id. Idempotent: if we
        already have results for this job_id (e.g. from a duplicate trigger), skip the API call."""
        if not job_id:
            raise PreventUpdate
        if isinstance(existing_results, dict) and existing_results.get("_job_id") == job_id:
            return no_update, no_update, no_update
        try:
            optimiser_results = optimiser_api.get_job_result(job_id)
        except Exception as e:
            logging.exception("Optimiser result fetch failed")
            optimiser_results = {
                "status": "error",
                "message": str(e),
            }
        optimiser_results["_job_id"] = job_id
        return optimiser_results, False, _loading_modal_default


# For standalone testing
if __name__ == "__main__":
    from dash import Dash
    external_stylesheets = [dbc.themes.DARKLY, dbc.icons.BOOTSTRAP]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = dbc.Container(form_layout(), fluid=True)
    register_callbacks(app)
    app.run(debug=True, port=8051)
