import os
import json
import logging
import time
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def custom_explore_gdf(
        gdf: gpd.GeoDataFrame,
        filepath: str,
        colour_col: str = None,
        open_in_browser: bool = True
        ):
    m = gdf.explore(column=colour_col)
    m.save(filepath)
    if open_in_browser:
        webbrowser.open(filepath)

def hex_grid(geometry, hex_diameter):
    """
    Create a hexagonal grid covering the input geometry.
    
    Parameters
    ----------
    geometry : shapely Polygon or MultiPolygon
    hex_diameter : float
        Diameter (flat-to-flat distance) of each hex cell in the same CRS units.
    """
    t0 = time.time()
    minx, miny, maxx, maxy = geometry.bounds
    dx = 3/2 * (hex_diameter / 2)
    dy = np.sqrt(3) * (hex_diameter / 2)

    # Generate hex centers
    x_coords = np.arange(minx - hex_diameter, maxx + hex_diameter, dx)
    y_coords = np.arange(miny - hex_diameter, maxy + hex_diameter, dy)

    hexes = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Offset every other column
            if i % 2 == 0:
                y0 = y + dy / 2
            else:
                y0 = y
            # Create hexagon centered at (x, y0)
            hexagon = Polygon([
                (x + hex_diameter/2 * np.cos(np.pi/3 * k),
                 y0 + hex_diameter/2 * np.sin(np.pi/3 * k))
                for k in range(6)
            ])
            # Keep only those intersecting the geometry
            if hexagon.intersects(geometry):
                hexes.append(hexagon)
    
    logger.info(f"Hex generation time: {round(time.time() - t0, 3)}s")
    return gpd.GeoDataFrame(geometry=hexes, crs="ESRI:102009")


def process_raster_data(raster_path, hex_gdf, value_column_name):
    """
    Process raster data and calculate mean values for each hex cell.
    
    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    hex_gdf : gpd.GeoDataFrame
        GeoDataFrame containing hex cell geometries.
    value_column_name : str
        Name for the calculated mean value column in the output.
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with hex cells and calculated mean values.
    """
    mean_values = []
    
    with rasterio.open(raster_path) as src:
        for _, hex in hex_gdf.iterrows():
            out_image, _ = mask(src, [hex.geometry], crop=True)
            out_image = out_image[0]  # first band
            out_image = np.where(out_image == src.nodata, np.nan, out_image)
            
            mean_val = np.nanmean(out_image)
            mean_values.append({
                "hex": hex["hex_id"],
                "geometry": hex.geometry,
                value_column_name: mean_val
            })
    
    return gpd.GeoDataFrame(mean_values, crs=hex_gdf.crs)


def main():
    """Main script execution."""
    states_path = r"C:\Users\Elis\Downloads\ne_110m_admin_1_states_provinces\ne_110m_admin_1_states_provinces.shp"
    us_states = gpd.read_file(states_path)
    us_states = us_states.to_crs("ESRI:102009")

    # remove Alaska and Hawaii
    us_states = us_states[~us_states["gn_name"].isin(["Alaska", "Hawaii"])]
    us_states = us_states[["gn_name", "geometry"]]
    us_states = us_states.rename(columns={"gn_name": "state"})

    # create hex cell representation
    usa_boundary = us_states.geometry.union_all()
    usa_hex = hex_grid(usa_boundary, hex_diameter=50000)
    usa_hex["hex_id"] = usa_hex.index

    # Process wind data
    logger.info("Processing wind speed data")
    wind_120_path = r"C:\Users\Elis\Downloads\us-wind-data\us-wind-data\wtk_conus_120m_mean_masked.tif"
    hex_wind_gdf = process_raster_data(
        raster_path=wind_120_path,
        hex_gdf=usa_hex,
        value_column_name="mean_120m_wind_speed"
        )

    # Process PV (GTI) data
    logger.info("Processing PV (GTI) data")
    gti_path = r"C:\Users\Elis\repos\us-datacentres\generate_hex_cell_data\inputs\GTI.tif"
    # Convert hex grid to EPSG:4326 for PV data
    usa_hex_4326 = usa_hex.to_crs("EPSG:4326")
    hex_gti_gdf = process_raster_data(
        raster_path=gti_path,
        hex_gdf=usa_hex_4326,
        value_column_name="mean_PV_GTI"
        )
    # Convert back to original CRS
    hex_gti_gdf = hex_gti_gdf.to_crs(usa_hex.crs)

    outputs_dir = "hex_cell_outputs"
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)

    # Save wind HTML map
    custom_explore_gdf(
        hex_wind_gdf,
        os.path.join(outputs_dir, "wind_map_all_hex.html"),
        colour_col="mean_120m_wind_speed"
        )

    # Save PV HTML map
    custom_explore_gdf(
        hex_gti_gdf,
        os.path.join(outputs_dir, "gti_map_all_hex.html"),
        colour_col="mean_PV_GTI"
        )

    # merge wnd and pv data on hex cell ids
    hex_wind_pv_gdf = hex_wind_gdf.merge(
        hex_gti_gdf.drop(columns=["geometry"]),
        on="hex", how="outer"
        )

    # create json lookup of state: hex ids (using wind data as reference)
    if not hex_wind_gdf.crs == us_states.crs:
        hex_wind_gdf = hex_wind_gdf.to_crs(us_states.crs)
    hex_states_df = hex_wind_gdf.sjoin(us_states)
    hex_states_df = hex_states_df[["hex", "state"]]
    state_hex_lookup = (
        hex_states_df.groupby("state")["hex"]
            .apply(list)
            .to_dict()
        )
    # save JSON lookup
    with open(os.path.join(outputs_dir, "states_hex_lookup.json"), "w") as f:
        json.dump(state_hex_lookup, f, indent=2, sort_keys=True)

    # Convert to EPSG:4326 and save all-US hex data
    hex_wind_gdf = hex_wind_gdf.to_crs(epsg=4326)
    hex_wind_gdf.to_file(
            os.path.join(outputs_dir, "all_hex_mean_wind_speeds.geojson")
            )

    hex_gti_gdf = hex_gti_gdf.to_crs(epsg=4326)
    hex_gti_gdf.to_file(
            os.path.join(outputs_dir, "all_hex_mean_PV_GTI.geojson")
            )

    hex_wind_pv_gdf = hex_wind_pv_gdf.to_crs(epsg=4326)
    hex_wind_pv_gdf.to_file(
            os.path.join(outputs_dir, "all_hex_mean_wind_and_PV_GTI.geojson")
            )

    logger.info("Script complete")


if __name__ == "__main__":
    main()