from math import floor, ceil
import streamlit as st
import plotly.express as px
from branca.colormap import linear
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import json
import plotly.graph_objects as go
from calculate_energy import solve
from pathlib import Path
from typing import Literal

st.set_page_config(layout="wide")
st.title("Data centre energy model")

def energy_calcs(slider_vals_dict):
    """
    """
    hex_cell = st.session_state.last_clicked_hex
    # get hex cell co-ordinates
    coords = get_hex_lat_long(hex_cell)

    # rename keys to match function if necessary
    arg_map = {
        "data_centre": "target_capacity",
        "wind": "wind_mw",
        "pv": "pv_mw",
        "bess_mw": "bess_mw",
        "bess_mwh": "bess_mwh",
    }
    kwargs = {arg_map[k]: v for k, v in slider_vals_dict.items()}

    st.session_state.energy_result, st.session_state.cost_result = solve(
        location=(coords["lat"], coords["long"], 100),
        mean_ws=11,
        rename_cols_for_data_centre_application=True,
        **kwargs
        )
    st.session_state.cost_result = st.session_state.cost_result / 1000

def store_slider_vals():
    """Store all slider values into session_state before calling energy_calcs"""
    st.session_state.slider_vals = {
        "data_centre": st.session_state.data_centre,
        "wind": st.session_state.wind,
        "pv": st.session_state.pv,
        "bess_mw": st.session_state.bess_mw,
        "bess_mwh": st.session_state.bess_mwh,
    }
    
    # Validate BESS sliders: both must be 0 or both must be positive
    bess_mw = st.session_state.bess_mw
    bess_mwh = st.session_state.bess_mwh
    
    if (bess_mw == 0 and bess_mwh > 0) or (bess_mw > 0 and bess_mwh == 0):
        st.session_state.bess_validation_error = True
        return  # Stop calculation
    else:
        st.session_state.bess_validation_error = False
    
    energy_calcs(st.session_state.slider_vals)  # pass the dictionary

@st.cache_data
def get_colormap(layer_type: Literal["wind", "pv"]):
    """Get colormap for given layer type."""
    print(f"Building {layer_type} colormap")
    if layer_type == "wind":
        vals = get_json_feature_range(wind_pv_dict, "mean_120m_wind_speed")
        colormap = linear.YlGn_09.scale(
            vals[0], vals[1]
        )
        colormap.caption = "Wind speed (120m)"
    elif layer_type == "pv":
        vals = get_json_feature_range(wind_pv_dict, "mean_PV_GTI")
        colormap = linear.YlOrBr_05.scale(
            vals[0], vals[1]
        )
        colormap.caption = "Global Horizontal Irradiance (W/m2)"
    else:
        return ValueError(f"{layer_type} not supported")
    return colormap

def create_colorbar_legend(colormap, num_ticks: int = 5) -> str:
    """Create HTML for a colorbar legend with tick marks.
    
    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to create a legend for.
    num_ticks : int, optional
        Number of tick marks to display (default: 5).
    
    Returns
    -------
    str
        HTML string for the legend.
    """
    min_val, max_val = colormap.vmin, colormap.vmax
    colors = [colormap(x) for x in colormap.index]
    
    # Generate intermediate values
    tick_values = [min_val + (max_val - min_val) * i / (num_ticks - 1) for i in range(num_ticks)]
    tick_spans = ' '.join([f'<span>{val:.1f}</span>' for val in tick_values])
    
    legend_html = f"""
    <div style="width: 100%; text-align: center; font-size: 14px;">
        <b>{colormap.caption}</b><br>
        <div style="height: 20px; width: 100%; background: linear-gradient(to right, {', '.join(colors)});"></div>
        <div style="display: flex; justify-content: space-between; width: 100%;">
            {tick_spans}
        </div>
    </div>
    """
    return legend_html

def get_folium_geojson(wind_pv_data, layer_type: str) -> folium.GeoJson:
    print(f"Preparing {layer_type} folium.GeoJson")
    cmap = get_colormap(
        layer_type="wind" if layer_select == "Wind speed" else "pv"
        )
    popup = folium.GeoJsonPopup(
        fields=["hex", "mean_120m_wind_speed", "mean_PV_GTI"],
        aliases=["Cell ID", "120m wind speed (m/s)", "GHI (W/m2)"],
        localize=True,
        labels=True
    )
    if layer_type == "Wind speed":
        return folium.GeoJson(
            wind_pv_data,
            name="state_wind_speeds",
            style_function=lambda feature: {
                "fillColor": (
                    cmap(feature["properties"]["mean_120m_wind_speed"])
                    if feature["properties"]["mean_120m_wind_speed"] is not None
                    else "#808080"
                    ),
                "color": "grey",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            highlight_function=lambda feature: {
                "weight": 3,
            },
            popup=popup
            )
    elif layer_type == "Global Horizontal Irradiance (GHI)":
        return folium.GeoJson(
            wind_pv_data,
            name="state_gtis",
            style_function=lambda feature: {
                "fillColor": cmap(feature["properties"]["mean_PV_GTI"]),
                "color": "grey",
                "weight": 1,
                "fillOpacity": 0.7,
            },
            highlight_function=lambda feature: {
                "weight": 3,
            },
            popup=popup
            )

@st.cache_data
def load_geojson_data() -> tuple[dict, pd.DataFrame]:
    """Load geojson data from hex_cell_outputs folder."""
    print("Loading geojson data")
    fpath = Path("hex_cell_outputs") / "all_hex_mean_wind_and_PV_GTI.geojson"
    with open(fpath) as f:
        data = json.load(f)
    df = gpd.read_file(fpath).drop(columns="geometry").set_index("hex")
    return data, df

@st.cache_data
def load_state_hex_lookup() -> dict[str, set]:
    print("Loading state-hex lookup")
    fpath = Path("hex_cell_outputs") / "states_hex_lookup.json"
    with open(fpath) as f:
        data = json.load(f)
    # convert list to set for faster searching
    data = {state: set(hex_array) for state, hex_array in data.items()}
    return data

@st.cache_data
def get_states_from_hex(
    states_hex_dict: dict[str, list[int]],
    target_hex: int
    ) -> list[str]:
    return [state for state, hex_array in states_hex_dict.items() if target_hex in hex_array]

def get_map_starting_parameters():
    try:
        zoom = st.session_state['hex_map']['zoom']
        x = st.session_state['hex_map']['center']["lng"]
        y = st.session_state['hex_map']['center']["lat"]
    except:
        x = -45
        y = 37.649034
        zoom = 4
    return {"x": x, "y": y, "zoom": zoom}

@st.cache_data
def get_json_feature_range(geojson: dict, property: str) -> list[int]:
    """
    TODO - docstring
    """
    props = (f["properties"].get(property) for f in geojson["features"])
    all_vals = [v for v in props if v is not None]
    return [min(all_vals), max(all_vals)]

@st.cache_data
def get_hex_lat_long(target_hex: int) -> dict[str, float]:
    geojson_crs = wind_pv_dict["crs"]["properties"]["name"]
    if geojson_crs[-5] != "CRS84":
        print(geojson_crs[-5])
        print("WARNING - expecting CRS84")
    features = wind_pv_dict["features"]
    hex_feature = [f for f in features if f["properties"]["hex"] == target_hex][0]
    co_ords_crs84 = hex_feature["geometry"]["coordinates"][0][0]
    # crs84 is in long, lat; wgs84 is in lat, long 
    # co_ord_wgs84 = [co_ords_crs84[1], co_ords_crs84[0]]
    return {"lat": co_ords_crs84[1], "long": co_ords_crs84[0]}

def create_1d_data_plot(
        df: pd.DataFrame,
        data_col: str,
        hex_id: int
        ):
    """
    df: dataframe containing wind speed and GTI data per hex cell
    data_col: dataframe column for plot values
    selected_hex: the celexted hex cell ID
    """

    if data_col == "mean_120m_wind_speed":
        color='rgb(36, 134, 68)'
        layer_name="Cell 120m wind speed (m/s)"
        x_range = [2, 11]  # TODO - make dynamic
    elif data_col == "mean_PV_GTI":
        color='rgb(215, 94, 13)'
        layer_name="Cell PV GHI (W/m2)"
        x_range = [1100, 2600]  # TODO - make dynamic
    else:
        raise ValueError(f"{data_col} not supported")
    
    min_val = df[data_col].min()
    max_val = df[data_col].max()

    fig = go.Figure()
    # vertical (line-ns) markers for the min/max values across US
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[0,0],
        mode='markers',
        marker=dict(
            symbol="line-ns",
            size=20,
            line=dict(
                color="grey",
                width=3
            )
        ),
        name="US min, max values"
    ))
    fig.add_trace(go.Scatter(
        x=[df.iloc[hex_id][data_col]],
        y=[0,0],
        mode='markers',
        marker_size=20,
        marker=dict(
            color=color,
        ),
        name=layer_name
    ))
    fig.update_xaxes(
        showgrid=True,
        range=x_range,
        ticks="inside",
        nticks=10 if data_col == "mean_120m_wind_speed" else None,  # TODO - tidy
        tick0=1 if data_col == "mean_120m_wind_speed" else None,
        dtick=1 if data_col == "mean_120m_wind_speed" else None
        )
    fig.update_yaxes(
        showgrid=False, 
        zeroline=True,
        zerolinecolor='grey',
        zerolinewidth=3,
        showticklabels=False)
    fig.update_layout(
        height=225,
        plot_bgcolor="rgb(26,28,36)"
        )
    return fig
    

# data preparation
wind_pv_dict, wind_pv_df = load_geojson_data()
state_hex_lookup = load_state_hex_lookup()

# session state management
if "last_clicked_hex" not in st.session_state:
    st.session_state.last_clicked_hex = ""

if "energy_result" not in st.session_state:
    st.session_state.energy_result = pd.DataFrame()

if "cost_result" not in st.session_state:
    st.session_state.cost_result = pd.DataFrame()

# custom styles
st.markdown(
    """<style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
        </style>
    """, unsafe_allow_html=True
)

with st.expander("1️⃣ Select location", expanded=True):
    map_col, graphs_col = st.columns(2)
    with map_col:

        layer_select = st.radio(
            "Map layer",
            options=["Wind speed", "Global Horizontal Irradiance (GHI)"],
            key="layer_radio",
            horizontal=True
        )

        print("Loading map")
        map_params = get_map_starting_parameters()
        m = folium.Map(
            location=[map_params["y"], map_params["x"]],
            zoom_start=map_params["zoom"]
            )
        get_folium_geojson(wind_pv_dict, layer_select).add_to(m)
        with st.form(key="map-form"):
            hex_map = st_folium(
                m,
                width=2000,
                height=500,
                key="hex_map",
                )
            # Build legend
            cmap = get_colormap(
                layer_type="wind" if layer_select == "Wind speed" else "pv"
                )
            legend_html = create_colorbar_legend(cmap, num_ticks=5)
            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            map_submitted = st.form_submit_button(
                label="Select hex cell",
                key="map-form-submit"
                )

        if map_submitted:
            new_hex = (
                hex_map
                and hex_map.get("last_active_drawing")
                and hex_map["last_active_drawing"]["properties"].get("hex")
            )
            if new_hex:
                st.session_state.last_clicked_hex = new_hex
            elif "last_clicked_hex" not in st.session_state:
                st.session_state.last_clicked_hex = None

    with graphs_col:
        if st.session_state.last_clicked_hex:
            selected_hex = st.session_state.last_clicked_hex
            # Titles
            st.markdown(f"### Cell attributes")
            st.markdown(f"#### Selected hex ID: {selected_hex}")
            states = get_states_from_hex(
                state_hex_lookup,
                selected_hex
                )
            states_text = "States" if len(states) > 1 else "State"
            st.markdown(f"#### {states_text}: {", ".join(states)}")

            # Wind speed data plot
            ws_fig = create_1d_data_plot(
                df=wind_pv_df,
                data_col="mean_120m_wind_speed",
                hex_id=selected_hex
            )
            st.plotly_chart(ws_fig, key="hex_ws_1d_plot")

            # PV data plot
            pv_fig = create_1d_data_plot(
                df=wind_pv_df,
                data_col="mean_PV_GTI",
                hex_id=selected_hex
            )
            st.plotly_chart(pv_fig, key="hex_pv_1d_plot")

        else:
            st.markdown(
                """
                <div style="
                    background-color:#172d43;
                    border-left: 0.25rem solid #2196f3;
                    padding: 0.75em 1em;
                    border-radius: 0.5em;
                    margin-bottom: 1em;">
                    <div style="line-height: 1.5em; font-size: 1.5em;">ℹ️</div>
                    <h3 style="color:#FFFFFF; margin:0;">Select a hex cell to view data</h3>
                    <p style="color:#FFFFFF; margin-top:0.5em; font-size:1em;">
                        It may take a couple of attempts - we have a little quirk of Streamlit to deal with...
                    </p>
                    <p style="color:#FFFFFF; margin-top:0.5em; font-size:1em;">
                        Once selected, move on to Stage 2 <i>(Select capacities)</i>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


with st.expander("2️⃣ Select capacities", expanded=True):
    with st.form(key="select_capacities_form"):
        st.markdown("### Inputs")
        st.slider("Data centre capacity (MW)", 0, 100, key="data_centre")
        st.slider("Wind capacity (MW)", 0, 100, key="wind")
        st.slider("Solar PV capacity (MW)", 0, 100, key="pv")
        bess1, bess2 = st.columns(2)
        with bess1:
            st.slider("Battery capacity (MW)", 0, 100, key="bess_mw")
        with bess2:
            st.slider("Battery capacity (MWh)", 0, 100, key="bess_mwh")

        if st.session_state.energy_result.empty:
            empty_results = True
        else:
            empty_results = False
        empty_results = st.session_state.energy_result.empty

        st.form_submit_button(
            "Calculate",
            key="select_capacities_form_submit",
            disabled=not st.session_state.last_clicked_hex,
            on_click=store_slider_vals  # store and call energy_calcs
        )
    
    # Display BESS validation error if present
    if st.session_state.get("bess_validation_error", False):
        st.error("⚠️ Both Battery capacity (MW) and Battery capacity (MWh) must be either both zero or both positive. Please adjust the values.")
    
    # with st.form(key="results"):
    with st.container(border=True, key="results_container"):
        st.markdown("### Results")

        if not st.session_state.energy_result.empty:

            total_costs_col, total_emissions_col = st.columns(2)

            with total_costs_col:
                with st.container(border = True):
                    lifetime_cost_total = round(st.session_state.cost_result["Lifetime cost"].sum())
                    st.markdown(
                        f"<h3 style='text-align: center;'>Lifetime cost: ${lifetime_cost_total:,}k</h3>",
                        unsafe_allow_html=True
                    )

            with total_emissions_col:
                with st.container(border = True):
                    print(st.session_state.cost_result.columns)
                    lifetime_co2_total = round(st.session_state.cost_result["Lifetime kgCO2"].sum())
                    st.markdown(
                        f"<h3 style='text-align: center;'>Lifetime emissions: {lifetime_co2_total:,} tCO2</h3>",
                        unsafe_allow_html=True
                    )

        tab1, tab2 = st.tabs(["Charts", "Full data tables"])

        with tab1:
            if not st.session_state.energy_result.empty:

                df = st.session_state.energy_result
                cols_to_plot = [
                    "Data centre demand",
                    "Wind generation",
                    "Solar PV generation",
                    "BESS power",
                    "Gas consumption"
                    ]
                df = df[["datetime"] + cols_to_plot]

                # Time-series plot
                df_long = df.melt(id_vars="datetime", var_name="variable", value_name="value")
                fig_ts = px.line(df_long, x="datetime", y="value", color="variable",
                    title="Modelled power flows")
                fig_ts.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Power (MW)",
                    template="plotly_white"
                )
                # Enable the range slider
                fig_ts.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(step="all", label="All"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                        ])
                    )
                )
                st.plotly_chart(fig_ts)
                color_map = {trace.name: trace.line.color for trace in fig_ts.data}

                annual_energy_col, costs_col = st.columns(2)
                
                with annual_energy_col:
                    # Annual energy plot
                    annual_sums = df[cols_to_plot].sum().reset_index()
                    annual_sums = annual_sums.round()
                    annual_sums.columns = ["variable", "total"]
                    show_ad = ["Wind generation", "Solar PV generation", "Gas consumption"]
                    annual_sums = annual_sums[annual_sums["variable"].isin(show_ad)]
                    fig_ad = px.bar(
                        annual_sums,
                        x="variable",
                        y="total",
                        color="variable",
                        text="total",
                        color_discrete_map=color_map,
                        labels={
                            "variable": "Energy source",
                            "total": "Annual energy genertaion / consumption (MWh)"
                            },
                        title="Annual energy genertaion / consumption"
                    )
                    fig_ad.update_traces(marker_line_width=0)
                    fig_ad.update_layout(template="plotly_white")
                    st.plotly_chart(fig_ad)

                with costs_col:
                    # cost plot
                    df_costs = st.session_state.cost_result
                    df_costs = df_costs.reset_index().rename(columns={"index": "tech"})
                    df_costs = df_costs.round()
                    # Melt into long format
                    df_costs_long = df_costs.melt(
                        id_vars="tech",
                        value_vars=["CAPEX", "20yr OPEX", "20yr fuel cost"],
                        var_name="Cost type",
                        value_name="cost"
                        )
                    # stacked bar chart
                    fig_costs = px.bar(
                        df_costs_long,
                        x="tech",
                        y="cost",
                        color="Cost type",
                        title="Lifetime cost by technology",
                        text="cost",
                        labels={
                            "tech": "Technology",
                            "cost": "USD (thousands)",
                        },
                        color_discrete_sequence=px.colors.qualitative.Safe
                        )
                    st.plotly_chart(fig_costs)

            else:
                st.markdown("### Hit 'Calculate' to view results")

        with tab2:
            if not st.session_state.energy_result.empty:
                st.markdown("#### Time series data")
                st.markdown("Demand / generation units are in MW")
                st.dataframe(st.session_state.energy_result)
                st.markdown("#### Cost and emissions data")
                st.markdown("Cost units: $k, Emission units: tCO2")
                df_costs = df_costs.rename(
                    columns={
                        "kgCO2": "tCO2",
                        "Lifetime kgCO2": "20yr tCO2"
                    }
                    )
                st.dataframe(df_costs)
            else:
                st.markdown("### Hit 'Calculate' to view results")

with st.expander("Methodology and assumptions"):
    pass