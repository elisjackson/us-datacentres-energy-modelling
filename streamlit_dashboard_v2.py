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

st.set_page_config(layout="wide")
st.title("Data centre energy model")

def get_central_loc(state):
    if state == "Washington":
        return [47, -108], 6
    elif state == "Oregon":
        return [44.37, -108], 6
    elif state == "California":
        return [36.94, -95.18], 5
    else:
        return [47, -108], 5

@st.cache_data
def get_colormap(layer_type: str):
    """Get colormap for given layer type."""
    print(f"Building {layer_type} colormap")
    if layer_type == "wind":
        vals = get_json_feature_range(wind_pv_dict, "mean_120m_wind_speed")
        colormap = linear.YlGn_09.scale(
            vals[0], vals[1]
        )
        colormap.caption = "Wind speed"
    elif layer_type == "pv":
        vals = get_json_feature_range(wind_pv_dict, "mean_PV_GTI")
        colormap = linear.YlOrBr_05.scale(
            vals[0], vals[1]
        )
        colormap.caption = "PV GTI"
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
    colormap = get_colormap(layer_type=layer_type)
    popup = folium.GeoJsonPopup(
        fields=["hex", "mean_120m_wind_speed", "mean_PV_GTI"],
        localize=True,
        labels=True
    )
    if layer_type == "wind":
        return folium.GeoJson(
            wind_pv_data,
            name="state_wind_speeds",
            style_function=lambda feature: {
                "fillColor": (
                    colormap(feature["properties"]["mean_120m_wind_speed"])
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
    elif layer_type == "pv":
        return folium.GeoJson(
            wind_pv_data,
            name="state_gtis",
            style_function=lambda feature: {
                "fillColor": colormap(feature["properties"]["mean_PV_GTI"]),
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
def load_geojson_data() -> (dict, pd.DataFrame):
    """Load geojson data from hex_cell_outputs folder."""
    print("Loading geojson data")
    fpath = r"hex_cell_outputs\all_hex_mean_wind_and_PV_GTI.geojson"
    with open(fpath) as f:
        data = json.load(f)
    df = gpd.read_file(fpath).drop(columns="geometry").set_index("hex")
    return data, df

@st.cache_data
def load_state_hex_lookup() -> dict:
    print("Loading state-hex lookup")
    fpath = r"hex_cell_outputs\states_hex_lookup.json"
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

# data preparation
wind_pv_dict, wind_pv_df = load_geojson_data()
state_hex_lookup = load_state_hex_lookup()

# session state management
if "last_clicked_hex" not in st.session_state:
    st.session_state.last_clicked_hex = ""

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
            options=["wind", "pv"],
            key="layer_radio",
            horizontal=True
        )

        print("Loading map")
        layer_colormaps = {
            "wind": get_colormap(layer_type="wind"),
            "pv": get_colormap(layer_type="pv")
        }
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
                key=f"hex_map",
                )
            # Build legend
            cmap = layer_colormaps[layer_select]
            legend_html = create_colorbar_legend(cmap, num_ticks=5)
            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            map_submitted = st.form_submit_button(
                label="Select hex cell",
                key="map-form-submit"
                )

        # --- Update session state safely ---
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
            st.markdown(f"### Selected hex ID: {selected_hex}")
            states = get_states_from_hex(
                state_hex_lookup,
                selected_hex
                )
            states_text = "States" if len(states) > 1 else "State"
            st.markdown(f"#### {states_text}: {", ".join(states)}")

            # Wind speed data plot
            st.markdown("#### Wind speed vs other US cells")
            min_ws = wind_pv_df["mean_120m_wind_speed"].min()
            max_ws = wind_pv_df["mean_120m_wind_speed"].max()
            ws_fig = go.Figure()
            ws_fig.add_trace(go.Scatter(
                x=[min_ws, max_ws], y=[0,0],
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
            ws_fig.add_trace(go.Scatter(
                x=[wind_pv_df.iloc[selected_hex]["mean_120m_wind_speed"]],
                y=[0,0],
                mode='markers',
                marker_size=20,
                marker=dict(
                    color='rgb(36, 134, 68)',
                ),
                name="Cell wind speed"
            ))
            ws_fig.update_xaxes(
                showgrid=True,
                range=[2, 11],
                ticks="inside",
                nticks=10,
                tick0=1, dtick=1
                )
            ws_fig.update_yaxes(
                showgrid=False, 
                zeroline=True,
                zerolinecolor='grey',
                zerolinewidth=3,
                showticklabels=False)
            ws_fig.update_layout(
                height=225,
                plot_bgcolor="rgb(26,28,36)"
                )
            st.plotly_chart(ws_fig, key="hex_ws")

            # PV data plot
            # st.markdown("#### PV GTI vs other US cells")
            min_gti = wind_pv_df["mean_PV_GTI"].min()
            max_gti = wind_pv_df["mean_PV_GTI"].max()
            print(min_gti)
            print(max_gti)
            pv_fig = go.Figure()
            pv_fig.add_trace(go.Scatter(
                x=[min_gti, max_gti], y=[0,0],
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
            pv_fig.add_trace(go.Scatter(
                x=[wind_pv_df.iloc[selected_hex]["mean_PV_GTI"]],
                y=[0,0],
                mode='markers',
                marker=dict(
                    color='rgb(215, 94, 13)',
                ),
                marker_size=20,
                name="Cell PV GTI"
            ))
            pv_fig.update_xaxes(
                showgrid=True,
                range=[1100, 2600],
                ticks="inside",
                # nticks=10,
                # tick0=1, dtick=1
                )
            pv_fig.update_yaxes(
                showgrid=False, 
                zeroline=True,
                zerolinecolor='grey',
                zerolinewidth=3,
                showticklabels=False)
            pv_fig.update_layout(
                height=225,
                plot_bgcolor="rgb(26,28,36)"
                )
            st.plotly_chart(pv_fig, key="hex_pv")

            st.markdown("#### Estimated capacity factors")
            st.markdown("**abc** 123")
        else:
            st.markdown("## Select a hex cell to view data")

with st.expander("2️⃣ Select capacities", expanded=False):
    with st.form(key="select_capacities_form"):
        wind_slider = st.slider("Wind MW", 0, 10)
        pv_slider = st.slider("PV MW", 0, 10)
        bess_mw_slider = st.slider("BESS MW", 0, 10)
        bess_mwh_slider = st.slider("BESS MWh", 0, 10)
        st.form_submit_button("Calculate")

with st.expander("3️⃣ View results", expanded=False):
    # df = pd.DataFrame()
    pass