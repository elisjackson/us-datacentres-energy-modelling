"""
Visualization functions for sanity checking meteo data.

This module provides functions to plot wind data for verification purposes.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def plot_data(
    gdf: gpd.GeoDataFrame,
    output_path: Optional[Union[str, Path]] = None,
    backend: str = "plotly",
    color_on: str = "mean_wind_speed",
    show: bool = True
) -> Optional[go.Figure]:
    """
    Plot data from GeoDataFrame for sanity checking.
    
    Args:
        gdf: GeoDataFrame with wind speed / irradiance data
        output_path: Optional path to save plot
        backend: "plotly" or "matplotlib"
        show: Whether to display the plot
        
    Returns:
        Plotly figure if backend is "plotly", None otherwise
    """
    if color_on not in gdf.columns:
        raise ValueError(f"GeoDataFrame must have '{color_on}' column")
    
    if backend == "plotly":
        return _plot_plotly(gdf, output_path, color_on, show)
    elif backend == "matplotlib":
        return _plot_matplotlib(gdf, output_path, show)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'plotly' or 'matplotlib'")


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
    
    fig = px.choropleth_mapbox(
        gdf_with_id,
        geojson=geojson,
        locations="id",
        color=color_on,
        color_continuous_scale="Viridis",
        mapbox_style="open-street-map",
        center={"lat": center_lat, "lon": center_lon},
        zoom=5,
        opacity=0.7,
        labels={color_on: color_on},
        title=f"Mean {color_on} Map"
    )
    
    fig.update_layout(
        margin=dict(r=0, t=30, l=0, b=0),
        height=600
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
    backend: str = "plotly",
    color_on: str = "mean_wind_speed"
) -> Optional[go.Figure]:
    """
    Load GeoJSON and create a sanity check plot.
    
    Args:
        geojson_path: Path to GeoJSON file
        output_plot_path: Optional path to save plot
        backend: "plotly" or "matplotlib"
        
    Returns:
        Plotly figure if backend is "plotly", None otherwise
    """
    geojson_path = Path(geojson_path)
    
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    logger.info(f"Loading GeoJSON: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    
    logger.info(f"GeoDataFrame shape: {gdf.shape}")
    logger.info(f"Columns: {gdf.columns.tolist()}")
    
    return plot_data(gdf, output_path=output_plot_path, backend=backend, color_on=color_on, show=True)
