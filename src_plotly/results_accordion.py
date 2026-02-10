import json
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

DEBUG_PRINT_TO_TERMINAL = True  # Set False to disable server-side print when Store updates

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
            tickfont=dict(color="#e0e0e0"),
            title_font=dict(color="#e0e0e0"),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(color="#e0e0e0"),
            title_font=dict(color="#e0e0e0"),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
            fixedrange=True,
        ),
        dragmode=False,
    )
    if show_placeholder:
        layout["annotations"] = [
            dict(
                text="Run the optimiser to see optimal capacities",
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

    if timeseries is None or not timeseries:
        fig.update_layout(**_fig_layout(xaxis_title="TODO", yaxis_title="TODO", show_placeholder=True))
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear",
            )
        )
        return fig

    for generator, values in timeseries.items():
        print(generator)
        print(values[:20])
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                name=generator,
                mode="lines",
            )
        )
    fig.update_layout(**_fig_layout(xaxis_title="TODO", yaxis_title="TODO", show_placeholder=False))
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear",
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

    fig.add_bar(
        x=list(capacities.keys()),
        y=list(capacities.values()),
        marker=dict(color="#78b4a0"),
    )
    fig.update_layout(
        **_fig_layout(
            xaxis_title="Generator / storage",
            yaxis_title="Capacity (MW)",
            show_placeholder=False,
        )
    )
    return fig

def create_costs_graph(costs: dict = None):
    """
    Create a grouped bar chart of costs: one group CAPEX, one group Marginal.
    costs must be {"capex": {carrier: value, ...}, "marginal": {carrier: value, ...}}.
    """
    # TODO - multiply by lifetime
    # TODO - un-annualise
    # TODO - check results
    # TODO - add OPEX
    # TODO - add CO2 cost
    fig = go.Figure()

    if costs is None or not costs:
        fig.update_layout(
            **_fig_layout(
                xaxis_title="Generator / storage",
                yaxis_title="Cost (€)",
                show_placeholder=True,
            )
        )
        return fig

    capex = costs.get("capex") or {}
    marginal = costs.get("marginal") or {}
    carriers = sorted(set(capex) | set(marginal))

    if not carriers:
        fig.update_layout(
            **_fig_layout(
                xaxis_title="Generator / storage",
                yaxis_title="Cost (€)",
                show_placeholder=True,
            )
        )
        return fig

    fig.add_trace(
        go.Bar(
            x=carriers,
            y=[capex.get(c, 0) for c in carriers],
            name="CAPEX",
            marker=dict(color="#78b4a0"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=carriers,
            y=[marginal.get(c, 0) for c in carriers],
            name="Marginal",
            marker=dict(color="#e0a86a"),
        )
    )
    fig.update_layout(
        **_fig_layout(
            xaxis_title="Generator / storage",
            yaxis_title="Cost (€)",
            show_placeholder=False,
        ),
        barmode="group",
        legend=dict(
            font=dict(color="#e0e0e0"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig

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
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            figure=create_optimal_capacities_graph(capacities=None),
                            id="optimal-capacities-fig",
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Graph(
                            figure=create_costs_graph(costs=None),
                            id="costs-fig",
                            config={"displayModeBar": False},
                        ),
                        md=6,
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
                        ),
                        md=12,
                    ),
                ]
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
    """Register callbacks for the results accordion (debug display + any future results UI)."""

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

        capex = {**generator_stats.get("capital_cost", {}), **storage_stats.get("capital_cost", {})}
        marginal = {**generator_stats.get("marginal_cost", {}), **storage_stats.get("marginal_cost", {})}

        all_costs = {"capex": capex, "marginal": marginal}

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