import json
import math
from typing import Literal
from datetime import datetime, timedelta
from pathlib import Path
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

DEBUG_PRINT_TO_TERMINAL = True  # Set False to disable server-side print when Store updates

DIR = Path(__file__).parent
COLOR_CONFIG_PATH = DIR / "config" / "chart_color_mapping.json"
DEFAULT_TECH_COLOR = "#95a5a6"


def _is_nan_key(k) -> bool:
    """True if key should be excluded from charts (NaN, None, or string 'nan')."""
    if k is None:
        return True
    if isinstance(k, float) and math.isnan(k):
        return True
    if isinstance(k, str) and str(k).strip().lower() == "nan":
        return True
    return False


def _drop_nan_keys(d: dict) -> dict:
    """Return a copy of the dict with NaN-like keys removed."""
    if not d:
        return d
    return {k: v for k, v in d.items() if not _is_nan_key(k)}


def _normalize_technology_key(name: str) -> str:
    if name is None:
        return ""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _load_technology_colors():
    fallback_color = DEFAULT_TECH_COLOR
    normalized_mapping = {}

    try:
        with open(COLOR_CONFIG_PATH, "r", encoding="utf-8") as f:
            color_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback_color, normalized_mapping

    configured_fallback = color_config.get("fallback")
    if isinstance(configured_fallback, str) and configured_fallback:
        fallback_color = configured_fallback

    technology_colors = color_config.get("technology_colors", {})
    if isinstance(technology_colors, dict):
        for tech_name, color in technology_colors.items():
            if not isinstance(tech_name, str) or not isinstance(color, str) or not color:
                continue
            normalized_mapping[_normalize_technology_key(tech_name)] = color

    return fallback_color, normalized_mapping


FALLBACK_TECH_COLOR, TECHNOLOGY_COLORS = _load_technology_colors()


def _get_technology_color(technology_name: str) -> str:
    normalized_name = _normalize_technology_key(technology_name)
    return TECHNOLOGY_COLORS.get(normalized_name, FALLBACK_TECH_COLOR)


def _adjust_hex_color(hex_color: str, amount: float) -> str:
    """
    Lighten/darken a hex colour by blending toward white/black.
    amount in [-1, 1]: positive -> lighter, negative -> darker.
    """
    if not isinstance(hex_color, str):
        return FALLBACK_TECH_COLOR

    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        return hex_color

    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return hex_color

    amount = max(-1.0, min(1.0, amount))
    if amount >= 0:
        r = round(r + (255 - r) * amount)
        g = round(g + (255 - g) * amount)
        b = round(b + (255 - b) * amount)
    else:
        factor = 1 + amount
        r = round(r * factor)
        g = round(g * factor)
        b = round(b * factor)

    return f"#{r:02x}{g:02x}{b:02x}"


def _fig_layout(xaxis_title: str, yaxis_title: str, show_placeholder=False):
    """Layout matching wind profile: dark theme, transparent background, same fonts/grid."""
    layout = dict(
        title="",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        title_font=dict(color="#e0e0e0"),
        xaxis=dict(
            tickfont=dict(color="#e0e0e0", size=10),
            title_font=dict(color="#e0e0e0", size=10),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(color="#e0e0e0", size=10),
            title_font=dict(color="#e0e0e0", size=10),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
            fixedrange=True,
        ),
        dragmode=False,
    )
    if show_placeholder:
        layout["annotations"] = [
            dict(
                text="No data",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#95a5a6", size=14),
            )
        ]
    return layout


def create_timeseries_plot(timeseries: dict = None):
    """
    Create a timeseries plot of the generation and storage capacities.

    timeseries shape:
    {
        "generator1": [100, 200, 300],
        "generator2": [150, 250, 350],
        "storage1": [100, 200, 300],
        "storage2": [150, 250, 350],
    }
    """
    fig = go.Figure()
    start_datetime = datetime(2025, 1, 1)

    x_axis_title = "Time"
    y_axis_title = "Generation / Storage Power (MW)"

    if timeseries is None or not timeseries:
        fig.update_layout(**_fig_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            show_placeholder=True
            ))
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
            )
        )
        return fig

    timeseries = _drop_nan_keys(timeseries)
    for generator, values in timeseries.items():
        fig.add_trace(
            go.Scatter(
                x=[start_datetime + timedelta(hours=hour) for hour in range(len(values))],
                y=values,
                name=generator,
                mode="lines",
                line=dict(color=_get_technology_color(generator)),
            )
        )
    fig.update_layout(**_fig_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        show_placeholder=False
        ))
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
        )
    )

    return fig


def create_optimal_capacities_graph(capacities: dict = None):
    """
    Create a bar chart of the optimal capacities for the generators and storage units.
    Styled to match the wind profile figure (dark theme, transparent bg, same fonts/grid).
    """
    fig = go.Figure()

    if capacities is None or not capacities:
        fig.update_layout(
            **_fig_layout(
                xaxis_title="Generator / storage",
                yaxis_title="Capacity (MW)",
                show_placeholder=True,
            )
        )
        return fig

    capacities = _drop_nan_keys(capacities)
    if not capacities:
        fig.update_layout(
            **_fig_layout(
                xaxis_title="Generator / storage",
                yaxis_title="Capacity (MW)",
                show_placeholder=True,
            )
        )
        return fig

    capacity_keys = list(capacities.keys())
    fig.add_bar(
        x=capacity_keys,
        y=list(capacities.values()),
        marker=dict(color=[_get_technology_color(key) for key in capacity_keys]),
    )
    fig.update_layout(
        **_fig_layout(
            xaxis_title="Generator / storage",
            yaxis_title="Capacity (MW)",
            show_placeholder=False,
        )
    )
    return fig


def create_annual_generation_graph(annual_generation: dict = None):
    """Create a bar chart of annual energy generated per generator."""
    fig = go.Figure()

    x_axis_title = "Generator"
    y_axis_title = "Annual generation (GWh)"

    if annual_generation is None or not annual_generation:
        fig.update_layout(
            **_fig_layout(
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                show_placeholder=False,
            )
        )
        fig.update_layout(
            annotations=[
                dict(
                    text="No data",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="#95a5a6", size=14),
                )
            ]
        )
        return fig

    annual_generation = _drop_nan_keys(annual_generation)
    if not annual_generation:
        fig.update_layout(
            **_fig_layout(
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                show_placeholder=False,
            )
        )
        fig.update_layout(
            annotations=[
                dict(
                    text="No data",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="#95a5a6", size=14),
                )
            ]
        )
        return fig

    generation_keys = list(annual_generation.keys())
    fig.add_bar(
        x=generation_keys,
        y=[value / 1000 for value in annual_generation.values()],
        marker=dict(color=[_get_technology_color(key) for key in generation_keys]),
    )
    fig.update_layout(
        **_fig_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            show_placeholder=False,
        )
    )
    return fig


def create_costs_graph(costs: dict = None):
    """
    Create a grouped bar chart of costs: one group CAPEX, one group Marginal.
    costs must be {"capex": {carrier: value, ...}, "marginal": {carrier: value, ...}}.
    """
    fig = go.Figure()

    x_axis_title = "Generator / storage"
    y_axis_title = "Annualised cost ($)"

    if costs is None or not costs:
        fig.update_layout(
            **_fig_layout(
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                show_placeholder=True,
            )
        )
        return fig

    capex = _drop_nan_keys(costs.get("capex") or {})
    opex = _drop_nan_keys(costs.get("opex") or {})
    energy_cost = _drop_nan_keys(costs.get("energy_cost") or {})
    co2_cost = _drop_nan_keys(costs.get("co2_cost") or {})
    carriers = list(capex.keys())

    if not carriers:
        fig.update_layout(
            **_fig_layout(
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                show_placeholder=True,
            )
        )
        return fig

    stack_shades = {
        "capex": -0.5,
        "opex": -0.2,
        "energy_cost": 0.2,
        "co2_cost": 0.5,
    }
    base_colors = {carrier: _get_technology_color(carrier) for carrier in carriers}

    fig.add_trace(
        go.Bar(
            x=carriers,
            y=[capex.get(c, 0) for c in carriers],
            name="CAPEX",
            marker=dict(
                color=[
                    _adjust_hex_color(base_colors[c], stack_shades["capex"])
                    for c in carriers
                ],
                pattern=dict(
                    shape="x"
                )
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=carriers,
            y=[opex.get(c, 0) for c in carriers],
            name="OPEX",
            marker=dict(
                color=[
                    _adjust_hex_color(base_colors[c], stack_shades["opex"])
                    for c in carriers
                ],
                pattern=dict(
                    shape="."
                )
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=carriers,
            y=[energy_cost.get(c, 0) for c in carriers],
            name="Energy cost",
            marker=dict(
                color=[
                    _adjust_hex_color(base_colors[c], stack_shades["energy_cost"])
                    for c in carriers
                ]
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=carriers,
            y=[co2_cost.get(c, 0) for c in carriers],
            name="CO₂ cost",
            marker=dict(
                color=[
                    _adjust_hex_color(base_colors[c], stack_shades["co2_cost"])
                    for c in carriers
                ]
            ),
        )
    )

    fig.update_layout(
        **_fig_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            show_placeholder=False,
        ),
        barmode="stack",
        legend=dict(
            font=dict(color="#e0e0e0"),
            yanchor="bottom",
            xanchor="left",
            y=0,
        ),
    )
    return fig

def create_card(
    header: str,
    value: float,
    unit: str = "",
    unit_location: Literal["left", "right"] = "right"
    ):
    value = round(value)
    # apply thousands separator
    value = f"{value:,}"
    if unit_location == "left":
        value_text = f"{unit} {value}" if unit else str(value)
    else:
        value_text = f"{value} {unit}" if unit else str(value)
    return dbc.Card(
        [
            dbc.CardHeader(header, className="results-card-header"),
            dbc.CardBody(
                [
                    html.H5(value_text, className="card-title results-card-value"),
                ]
            )
        ],
        className="results-summary-card",
    )


def results_layout():
    return html.Div(
        [
            dcc.Store(id="optimiser-results-data"),  # In-memory only; storage_type="local" can cause "Maximum update depth exceeded" with dependent callbacks
            dbc.Row(
                [
                    dbc.Col(create_card("Data centre capacity", 1000), id="data-centre-capacity-card"),
                    dbc.Col(create_card("Total generation capacity", 1000), id="total-generation-capacity-card"),
                    dbc.Col(create_card("Annualised cost", 1000), id="total-cost-card"),
                    dbc.Col(create_card("Total emissions", 10000), id="total-emissions-card"),
                ],
                className="align-items-center gy-3",
            ),
            html.P(
                "All costs are in 2023 USD.",
                className="mt-3 mb-2 text-note-white",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(
                                figure=create_optimal_capacities_graph(capacities=None),
                                id="optimal-capacities-fig",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "100%"},
                            ),
                            className="results-bar-chart-wrapper",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            dcc.Graph(
                                figure=create_annual_generation_graph(annual_generation=None),
                                id="annual-generation-fig",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "100%"},
                            ),
                            className="results-bar-chart-wrapper",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            dcc.Graph(
                                figure=create_costs_graph(costs=None),
                                id="costs-fig",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "100%"},
                            ),
                            className="results-bar-chart-wrapper",
                        ),
                        md=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            figure=create_timeseries_plot(timeseries=None),
                            id="timeseries-fig",
                            config={"displayModeBar": False},
                            style={"height": "300px"},
                        ),
                        md=12,
                    ),
                ]
            ),
        ],
        className="p-3 results-layout",
        style={
            "backgroundColor": "rgb(5, 13, 24)",
            "borderRadius": "8px",
            "border": "1px solid rgb(111, 111, 111)",
            "width": "100%",
        },
    )


def register_callbacks(app):
    """Register callbacks for the results accordion."""

    @app.callback(
        Output("timeseries-fig", "figure"),
        Input("optimiser-results-data", "data"),
    )
    def _update_timeseries_plot(data):

        if data is None:
            return create_timeseries_plot(timeseries=None)

        generation_timeseries = data.get("generation_ts", {})
        storage_timeseries = data.get("storage_ts", {})
        timeseries = {**generation_timeseries, **storage_timeseries}

        return create_timeseries_plot(timeseries=timeseries)

    @app.callback(
        Output("costs-fig", "figure"),
        Input("optimiser-results-data", "data"),
    )
    def _update_costs_graph(data):

        if data is None:
            return create_costs_graph(costs=None)

        generator_stats = data.get("generator_stats", {})
        storage_stats = data.get("storage_stats", {})

        capex = {**generator_stats.get("total_capex", {}), **storage_stats.get("total_capex", {})}
        opex = {**generator_stats.get("total_opex", {}), **storage_stats.get("total_opex", {})}
        energy_cost = {**generator_stats.get("total_energy_cost", {}), **storage_stats.get("total_energy_cost", {})}
        co2_cost = {**generator_stats.get("total_co2_cost", {}), **storage_stats.get("total_co2_cost", {})}

        all_costs = {"capex": capex, "opex": opex, "energy_cost": energy_cost, "co2_cost": co2_cost}

        return create_costs_graph(costs=all_costs)

    @app.callback(
        Output("optimal-capacities-fig", "figure"),
        Input("optimiser-results-data", "data"),
    )
    def _update_optimal_capacities_graph(data):

        if data is None:
            return create_optimal_capacities_graph(capacities=None)

        generator_capacities = data.get("generator_stats", {}).get("p_nom_opt", {})
        storage_capacities = data.get("storage_stats", {}).get("p_nom_opt", {})
        all_capacities = {**generator_capacities, **storage_capacities}

        return create_optimal_capacities_graph(capacities=all_capacities)

    @app.callback(
        Output("annual-generation-fig", "figure"),
        Input("optimiser-results-data", "data"),
    )
    def _update_annual_generation_graph(data):

        if data is None:
            return create_annual_generation_graph(annual_generation=None)

        annual_generation = data.get("annual_generation", {})

        return create_annual_generation_graph(annual_generation=annual_generation)

    @app.callback(
        [
            Output("data-centre-capacity-card", "children"),
            Output("total-generation-capacity-card", "children"),
            Output("total-cost-card", "children"),
            Output("total-emissions-card", "children"),
        ],
        Input("optimiser-results-data", "data"),
    )
    def _update_summary_cards(data):
        if data is None:
            return (
                create_card("Data centre capacity", 0, "MW"),
                create_card("Total generation capacity", 0, "MW"),
                create_card("Annualised cost", 0, "$", unit_location="left"),
                create_card("Total emissions", 0, "tCO₂"),
            )
        return (
            create_card("Data centre capacity", data.get("load", 0), "MW"),
            create_card("Total generation capacity", data.get("total_generation_capacity", 0), "MW"),
            create_card("Annualised cost", data.get("total_cost", 0), "$", unit_location="left"),
            create_card("Total emissions", data.get("total_emissions", 0), "tCO₂"),
        )
