"""
Visualization functions for sanity checking meteo data.

This module provides functions to plot wind data for verification purposes.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Union, Optional, Any
import logging

logger = logging.getLogger(__name__)


def plot_data(
    gdf: gpd.GeoDataFrame,
    output_path: Optional[Union[str, Path]] = None,
    backend: str = "explore",
    color_on: str = "mean_wind_speed",
    show: bool = False
) -> Optional[Any]:
    """
    Plot data from GeoDataFrame for sanity checking.

    Args:
        gdf: GeoDataFrame with wind speed / irradiance data
        output_path: Optional path to save plot (HTML for explore, PNG for matplotlib)
        backend: "explore" (folium map, default), "plotly", or "matplotlib"
        color_on: Column name to color the map by
        show: Whether to display the plot in a browser (explore/plotly)

    Returns:
        folium Map if backend is "explore", Plotly figure if "plotly", None if "matplotlib"
    """
    if color_on not in gdf.columns:
        raise ValueError(f"GeoDataFrame must have '{color_on}' column")

    if backend == "explore":
        return _plot_explore(gdf, output_path, color_on, show)
    elif backend == "plotly":
        return _plot_plotly(gdf, output_path, color_on, show)
    elif backend == "matplotlib":
        return _plot_matplotlib(gdf, output_path, show)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'explore', 'plotly' or 'matplotlib'")


def _plot_explore(
    gdf: gpd.GeoDataFrame,
    output_path: Optional[Union[str, Path]],
    color_on: str,
    show: bool
) -> Any:
    """Create interactive folium map with gdf.explore() and save to HTML."""
    gdf_web = gdf.to_crs(epsg=4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf.copy()
    col = gdf_web[color_on]
    is_categorical = col.dtype == bool or (getattr(col.dtype, "name", "") == "category") or col.nunique() <= 10

    if is_categorical:
        unique_vals = col.dropna().unique().tolist()
        if set(unique_vals) <= {True, False}:
            # False=blue (offshore), True=green (onshore)
            cmap = mcolors.ListedColormap(["#3498db", "#2ecc71"])
        else:
            colors_list = plt.cm.Set1([i % 9 / 9 for i in range(len(unique_vals))])
            cmap = mcolors.ListedColormap(colors_list)
        m = gdf_web.explore(
            column=color_on,
            categorical=True,
            cmap=cmap,
            tiles="CartoDB positron",
            tooltip=True,
            popup=[color_on],
            legend=True,
        )
    else:
        m = gdf_web.explore(
            column=color_on,
            cmap="Viridis",
            tiles="CartoDB positron",
            tooltip=True,
            popup=[color_on],
            legend=True,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info(f"Map saved to: {output_path}")
        if show:
            import webbrowser
            webbrowser.open(output_path.as_uri())
    elif show:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            m.save(f.name)
            import webbrowser
            webbrowser.open(f"file://{f.name}")

    return m


def _plot_plotly(
    gdf: gpd.GeoDataFrame,
    output_path: Optional[Union[str, Path]],
    color_on: str,
    show: bool
) -> go.Figure:
    """Create Plotly choropleth map."""
    # Convert GeoDataFrame to GeoJSON format for Plotly
    import json
    geojson = json.loads(gdf.to_json())
    
    # Create a unique ID column for matching
    gdf_with_id = gdf.copy()
    gdf_with_id["id"] = gdf_with_id.index
    
    # Calculate center for map
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Use geo-based px.choropleth (no tile loading) so saved HTML works when opened locally
    # Flat 2D projection (equirectangular) so it doesn't look like a globe
    col = gdf_with_id[color_on]
    is_categorical = col.dtype == bool or (col.dtype.name == "category") or col.nunique() <= 10
    common_kw = dict(
        geojson=geojson,
        locations="id",
        featureidkey="id",
        center={"lat": center_lat, "lon": center_lon},
        fitbounds="geojson",
        projection="equirectangular",
        scope="world",
        labels={color_on: color_on},
    )
    if is_categorical:
        unique_vals = col.dropna().unique().tolist()
        if set(unique_vals) <= {True, False}:
            color_discrete_map = {True: "#2ecc71", False: "#3498db"}
        else:
            color_discrete_map = {v: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                                 for i, v in enumerate(unique_vals)}
        fig = px.choropleth(
            gdf_with_id,
            color=color_on,
            color_discrete_map=color_discrete_map,
            title=f"{color_on} Map",
            **common_kw,
        )
    else:
        fig = px.choropleth(
            gdf_with_id,
            color=color_on,
            color_continuous_scale="Viridis",
            title=f"Mean {color_on} Map",
            **common_kw,
        )

    fig.update_traces(marker_opacity=0.7, selector=dict(type="choropleth"))
    fig.update_layout(
        margin=dict(r=0, t=30, l=0, b=0),
        height=600,
    )
    # Keep geo flat and minimal: equirectangular, simple coastlines, no fancy basemap
    fig.update_geos(
        projection_type="equirectangular",
        showcoastlines=True,
        showland=True,
        showocean=True,
        landcolor="rgb(240, 240, 240)",
        oceancolor="rgb(204, 229, 255)",
        coastlinecolor="rgb(128, 128, 128)",
        fitbounds="geojson",
    )
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Plot saved to: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def _plot_matplotlib(
    gdf: gpd.GeoDataFrame,
    output_path: Optional[Union[str, Path]],
    show: bool
) -> None:
    """Create matplotlib plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gdf.plot(
        column="mean_wind_speed",
        ax=ax,
        legend=True,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.1
    )
    
    ax.set_title("Mean Wind Speed Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def sanity_check_geojson(
    geojson_path: Union[str, Path],
    output_plot_path: Optional[Union[str, Path]] = None,
    backend: str = "explore",
    color_on: str = "mean_wind_speed"
) -> Optional[Any]:
    """
    Load GeoJSON and create a sanity check plot.

    Args:
        geojson_path: Path to GeoJSON file
        output_plot_path: Optional path to save plot (HTML for explore, PNG for matplotlib)
        backend: "explore" (default), "plotly", or "matplotlib"
        color_on: Column name to color the map by

    Returns:
        folium Map if backend is "explore", Plotly figure if "plotly", None otherwise
    """
    geojson_path = Path(geojson_path)
    
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    logger.info(f"Loading GeoJSON: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    
    logger.info(f"GeoDataFrame shape: {gdf.shape}")
    logger.info(f"Columns: {gdf.columns.tolist()}")
    logger.info(f"Color on: {color_on}")
    
    return plot_data(gdf, output_path=output_plot_path, backend=backend, color_on=color_on, show=False)
