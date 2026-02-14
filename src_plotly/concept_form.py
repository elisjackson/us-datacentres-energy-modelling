import dash_bootstrap_components as dbc
import json
import pandas as pd
from dash import Dash, Input, Output, State, html, dcc, ctx, no_update
from dash.dependencies import ALL, MATCH
from typing import List, Literal
from pathlib import Path

external_stylesheets = [dbc.themes.DARKLY, dbc.icons.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

DIR = Path(__file__).parent
CONFIG_DIR = DIR / "config"

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
            self.cost_choice_params[subtype] = self.cost_choices_df[
                self.cost_choices_df["subtype"] == subtype
            ]["cost_parameter"].unique()
            
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
                className="d-flex align-items-center",
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
                    dbc.Col(html.Label(row["cost_parameter"]), width=3),
                    dbc.Col(html.Label(row["value"]), width=3),
                    dbc.Col(html.Label(row["unit"]), width=3),
                    dbc.Col(html.Label(row["source"]), width=3),
                ])
            )
        return html.Div([
            dbc.Row([
                dbc.Col(html.Label("Additional Assumptions"), width=12)
            ]),
            *dbc_rows  # Unpack the list of Rows
        ])

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
        return pd.DataFrame(rows)

    def build_all_cost_columns(self, subtype: str) -> List[dbc.Row]:
        """Build all cost columns for the given subtype."""
        return [self.build_cost_column(param, subtype) for param in self.cost_choice_params[subtype]]

    def _create_cost_level_row(self, cost_parameter: str, level: str, value=None, is_active=False, input_disabled=False):
        """Helper to create a button + input row for a cost level."""
        # Style for inputs: different styling for selected vs unselected disabled inputs
        base_style = {
            "borderTopLeftRadius": "0",
            "borderBottomLeftRadius": "0",
            "borderTopRightRadius": "8px",
            "borderBottomRightRadius": "8px",
        }
        
        if input_disabled:
            if is_active:
                # Selected disabled input: less faded, highlighted background
                input_style = {
                    **base_style,
                    "opacity": "0.85",
                    "cursor": "not-allowed",
                    "backgroundColor": "#2c3e50",
                    "fontWeight": "600",
                }
            else:
                # Unselected disabled input: more faded
                input_style = {
                    **base_style,
                    "opacity": "0.4",
                    "cursor": "not-allowed",
                    "backgroundColor": "#1a1a1a",
                }
        else:
            # Enabled input (Custom when active)
            input_style = base_style
        
        button_style = {
            "borderTopLeftRadius": "8px",
            "borderBottomLeftRadius": "8px",
            "borderTopRightRadius": "0",
            "borderBottomRightRadius": "0",
        }
        
        return dbc.Row([
            dbc.Col(
                dbc.Button(
                    level,
                    id={"type": "cost-level-btn", "card": self.card_id, "param": cost_parameter, "level": level},
                    className="w-100",
                    color="primary",
                    outline=not is_active,
                    active=is_active,
                    style=button_style,
                ),
                width=self.BTN_COL_WIDTH,
            ),
            dbc.Col(
                dbc.Input(
                    type="number",
                    value=value,
                    placeholder="0" if value is not None else "Custom value",
                    disabled=input_disabled,
                    style=input_style,
                    id={"type": "cost-value-input", "card": self.card_id, "param": cost_parameter, "level": level},
                ),
                width=self.INPUT_COL_WIDTH,
            ),
        ], className="g-0")
    
    def build_cost_column(self, cost_parameter: str, selected_subtype: str):
        """
        Build one cost column, filtered by subtype and cost parameter.

        E.g. CAPEX column containing a title, and a row for each cost level.
        """
        df = self.cost_choices_df.loc[
            (self.cost_choices_df["subtype"] == selected_subtype) &
            (self.cost_choices_df["cost_parameter"] == cost_parameter)
        ]
        print(f"Building {cost_parameter} for {selected_subtype}:")
        print(df)

        unit = df["unit"].unique()[0]

        # Build a row for each cost level (first one is active by default)
        level_rows = []
        for i, row in df.iterrows():
            level_rows.append(
                self._create_cost_level_row(
                    cost_parameter=cost_parameter,
                    level=row["level"],
                    value=row["value"],
                    is_active=(i == 0),  # First row is active
                    input_disabled=True  # Pre-defined values are read-only
                )
            )
        
        # Append a customizable row (enabled only when Custom button is clicked)
        level_rows.append(
            self._create_cost_level_row(
                cost_parameter=cost_parameter,
                level="Custom",
                value=None,
                is_active=False,
                input_disabled=True  # Starts disabled; callback enables when Custom button clicked
            )
        )
        
        # Label row on top, then level rows below
        return html.Div([
            dbc.Row([
                dbc.Col(html.Label(cost_parameter, className="fw-bold"), width=self.BTN_COL_WIDTH),
                dbc.Col(html.Label(unit), width=self.INPUT_COL_WIDTH),
            ], className="mb-2"),
            *level_rows  # Unpack the list of Rows
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
            className="mb-3",
        )

    def create_card_body(self):
        children = []
        
        # Subtype pill buttons row
        if self.subtype_buttons is not None:
            children.append(
                dbc.Row([
                    dbc.Col(html.Label("Select base cost assumptions:"), width="auto"),
                    dbc.Col(self.subtype_buttons, width="auto")
                ], className="align-items-center mb-3")
                )
        
        # Add cost columns in a container that can be updated dynamically
        children.append(
            html.Div(
                dbc.Row([
                    dbc.Col(c) for c in self.cost_columns[self.default_subtype]
                ], className="mb-3"),
                id={"type": "cost-columns-container", "card": self.card_id}
            )
        )
        
        return dbc.CardBody(children, className="py-3")


# read generation form from config/generation_form.json
with open(CONFIG_DIR / "generation_form copy.json", "r") as f:
    GENERATION_CONFIG = json.load(f)


wind_card = GenerationInput("Wind")
solar_card = GenerationInput("Solar", initial_open=True)
gas_card = GenerationInput("Gas")
smr_card = GenerationInput("SMR")
grid_electricity_card = GenerationInput("Grid Electricity")
battery_storage_card = GenerationInput("Battery Storage")
co2_card = GenerationInput("CO2")
generation_cards = [solar_card, wind_card, gas_card, smr_card, grid_electricity_card]
generation_card_ids = [card.card_id for card in generation_cards]
# Store cards by ID for callback access (include battery storage and CO2 for callbacks)
generation_cards_dict = {card.card_id: card for card in generation_cards}
generation_cards_dict[battery_storage_card.card_id] = battery_storage_card
generation_cards_dict[co2_card.card_id] = co2_card

app.layout = dbc.Container(
    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.H1("Concept Form"), width=12),
                ],
            ),
            # Selection controls row
            dbc.Row(
                [
                    # Generation selection column
                    dbc.Col(
                        [
                            dbc.Label("Select generation / energy sources", className="mb-2 text-center"),
                            html.Div(
                                [
                                    dbc.Button(
                                        card.title,
                                        id={"type": "generation-pill", "index": card.card_id},
                                        color="primary",
                                        outline=False,
                                        active=True,
                                        className="rounded-pill me-2 px-3",
                                    )
                                    for card in generation_cards
                                ],
                                className="d-flex align-items-center justify-content-center",
                            ),
                        ],
                        width="auto",
                    ),
                    # Battery storage column
                    dbc.Col(
                        [
                            dbc.Label("Battery storage", className="mb-2 text-center"),
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
                        className="ps-4",
                    ),
                    # Carbon price column
                    dbc.Col(
                        [
                            dbc.Label("Include Carbon price", className="mb-2 text-center"),
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
                        className="ps-4",
                    ),
                ],
                className="mb-3 align-items-start",
            ),
            dbc.Row(
                [
                    html.Div(
                        dbc.Col(
                            card.collapsible_card,
                            width=12,
                        ),
                        id={"type": "generation-card-col", "index": card.card_id},
                        className="card-container-animated card-visible",
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
            # Optimise button at the bottom
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Optimise",
                            id="optimise-button",
                            color="primary",
                            className="w-100 mt-4",
                            disabled=False,  # Will be controlled by callback
                        ),
                        width=12,
                    )
                ],
            ),
        ],
    ),
    style={
        "backgroundColor": "rgba(255,255,255,0.06)",
        "borderRadius": "8px",
        "width": "100%",
    },
)


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
    prevent_initial_call=True,
)
def toggle_subtype_pills_and_update_columns(n_clicks_list, pill_ids):
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
    
    # Update active states: only clicked pill is active
    active_states = [pill["subtype"] == selected_subtype for pill in pill_ids]
    outline_states = [not active for active in active_states]
    
    # Rebuild cost columns for the new subtype
    cost_columns = card.build_all_cost_columns(selected_subtype)
    
    # Add the additional assumptions column for this subtype (if it exists)
    if card.has_cost_assumptions:
        cost_columns.append(card.additional_assumptions_column[selected_subtype])
    
    # Return the updated states and columns
    columns_row = dbc.Row([
        dbc.Col(c) for c in cost_columns
    ], className="mb-3")
    
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
    Output({"type": "cost-level-btn", "card": ALL, "param": ALL, "level": ALL}, "active"),
    Output({"type": "cost-level-btn", "card": ALL, "param": ALL, "level": ALL}, "outline"),
    Output({"type": "cost-value-input", "card": ALL, "param": ALL, "level": ALL}, "disabled"),
    Output({"type": "cost-value-input", "card": ALL, "param": ALL, "level": ALL}, "style"),
    Input({"type": "cost-level-btn", "card": ALL, "param": ALL, "level": ALL}, "n_clicks"),
    State({"type": "cost-level-btn", "card": ALL, "param": ALL, "level": ALL}, "active"),
    prevent_initial_call=True,
)
def toggle_cost_level_buttons(n_clicks_list, current_active_states):
    """Make only the clicked button active within its cost parameter group (like radio buttons).
    Also enable Custom input only when Custom button is selected."""
    from dash import callback_context
    
    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update, no_update, no_update
    
    # Get which button was clicked
    clicked_card = triggered["card"]
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
        # Check if this button is in the same (card, param) group as the clicked button
        same_group = (btn_id["card"] == clicked_card and btn_id["param"] == clicked_param)
        
        if same_group:
            # Within the clicked group: only the clicked button is active
            is_clicked_button = (btn_id["level"] == clicked_level)
            active_states.append(is_clicked_button)
            outline_states.append(not is_clicked_button)
        else:
            # Different group: preserve current state
            active_states.append(current_active_states[i])
            outline_states.append(not current_active_states[i])
    
    # Build disabled states and styles for all inputs
    input_disabled_states = []
    input_styles = []
    
    base_style = {
        "borderTopLeftRadius": "0",
        "borderBottomLeftRadius": "0",
        "borderTopRightRadius": "8px",
        "borderBottomRightRadius": "8px",
    }
    
    selected_disabled_style = {
        **base_style,
        "opacity": "0.85",
        "cursor": "not-allowed",
        "backgroundColor": "#2c3e50",
        "fontWeight": "600",
    }
    
    unselected_disabled_style = {
        **base_style,
        "opacity": "0.4",
        "cursor": "not-allowed",
        "backgroundColor": "#1a1a1a",
    }
    
    for inp_id in input_ids:
        # Custom inputs are enabled only if their corresponding Custom button is active
        if inp_id["level"] == "Custom":
            # Find the corresponding Custom button's active state
            custom_btn_active = False
            for i, btn_id in enumerate(button_ids):
                if (btn_id["card"] == inp_id["card"] and 
                    btn_id["param"] == inp_id["param"] and 
                    btn_id["level"] == "Custom"):
                    custom_btn_active = active_states[i]
                    break
            is_disabled = not custom_btn_active
            input_disabled_states.append(is_disabled)
            input_styles.append(unselected_disabled_style if is_disabled else base_style)
        else:
            # Non-custom inputs are always disabled - check if their button is active
            btn_is_active = False
            for i, btn_id in enumerate(button_ids):
                if (btn_id["card"] == inp_id["card"] and 
                    btn_id["param"] == inp_id["param"] and 
                    btn_id["level"] == inp_id["level"]):
                    btn_is_active = active_states[i]
                    break
            input_disabled_states.append(True)
            # Selected inputs get highlighted even though disabled
            input_styles.append(selected_disabled_style if btn_is_active else unselected_disabled_style)
    
    return active_states, outline_states, input_disabled_states, input_styles


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
    Input({"type": "generation-pill", "index": ALL}, "active"),
)
def update_optimise_button(pill_active_states):
    """Enable Optimise button only if at least one generation source is selected."""
    # Check if at least one pill is active
    has_active = any(pill_active_states)
    is_disabled = not has_active
    
    return is_disabled


if __name__ == "__main__":
    app.run(debug=True, port=8051)
