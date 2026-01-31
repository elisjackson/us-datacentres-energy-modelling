"""Get bbox of a specified country from geoBoundariesCGAZ_ADM0.gpkg"""

import geopandas as gpd
import os
from explore_gpkg import save_map
import logging
from typing import Literal
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = r"C:\Users\Elis\repos\us-datacentres\data\downloads\shapefiles"
OUTPUT_DIR = r"C:\Users\Elis\repos\us-datacentres\data\processed"
BUFFER_DISTANCE = 200000  # in meters
AREA_THRESHOLD = 5e8  # in m2
MAX_POLYGONS_THRESHOLD = 2
COUNTRY_CONFIG = {
    "United Kingdom": {
        "filename": "Countries_December_2024_Boundaries_UK_BUC_-4247675800557514417.gpkg",
        "layer": "CTRY_DEC_2024_UK_BUC",
        "crs_epsg": 27700,
        "extract_country_required": False,
        "filter_to_main_territory": False
    },
    "Canada": {
        "filename": "geoBoundariesCGAZ_ADM0.gpkg",
        "layer": "globalADM0",
        "crs_epsg": 4326,
        "extract_country_required": True,
        "filter_to_main_territory": False
    },
    "Other": {
        "filename": "geoBoundariesCGAZ_ADM0.gpkg",
        "layer": "globalADM0",
        "crs_epsg": 4326,
        "extract_country_required": True,
        "filter_to_main_territory": True
    }
}

def fill_holes(geom):
    """
    Remove interior rings (holes) from a polygon or multipolygon.
    Returns a new geometry with only the exterior boundary.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        if geom.interiors:
            return Polygon(geom.exterior)
        return geom
    if geom.geom_type == "MultiPolygon":
        filled = [Polygon(p.exterior) if p.interiors else p for p in geom.geoms]
        return MultiPolygon(filled)
    return geom


def add_buffer(gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
    """
    Add buffer to a geometry
    Args:
        gdf: GeoDataFrame with geometry column
        buffer_distance: Distance to buffer in meters
    Returns:
        GeoDataFrame with buffered geometry
    """
    logger.info("Adding buffer")
    gdf_m = gdf.to_crs(epsg=27700)
    gdf_m["geometry"] = gdf_m.buffer(buffer_distance)
    gdf_buffered = gdf_m.to_crs(epsg=4326)
    logger.info("Buffer complete")
    return gdf_buffered

def extract_geometry_by_country(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract geometry of a specified country from geoBoundariesCGAZ_ADM0.gpkg"""

    logger.info(f"Extracting geometry for {country}")
    gdf = gdf[gdf["shapeName"] == country]

    if gdf.empty:
        raise ValueError(f"No features found for shapeName == {country}")

    return gdf


def multipolygon_to_polygons(gdf: gpd.GeoDataFrame) -> gpd.geodataframe:
    """
    Take a multipolygon, disaggregate to polygons
    """
    gdf_e = gdf.to_crs(epsg=3857)
    gdf_e = gdf_e.explode()
    gdf_e["area"] = gdf_e.geometry.area
    logger.info(f"Min area: {gdf_e["area"].min()}")
    logger.info(f"Max area: {gdf_e["area"].max()}")
    gdf_e = gdf_e.to_crs(epsg=4326)
    return gdf_e

def filter_to_main_territory(
    gdf: gpd.GeoDataFrame,
    method: Literal["area", "largest_polygons"]
    ) -> gpd.GeoDataFrame:
    """
    """
    if method == "area":
        gdf = gdf[gdf["area"] > AREA_THRESHOLD]
    elif method == "largest_polygons":
        gdf = gdf.sort_values(by="area", ascending=False)
        gdf = gdf.head(MAX_POLYGONS_THRESHOLD)
    return gdf


def fix_geometry(geom):
    """Fix a single geometry if invalid."""
    # nonlocal invalid_count, empty_count

    if geom is None or geom.is_empty:
        # empty_count += 1
        return None

    if not geom.is_valid:
        # invalid_count += 1
        # Try to fix with make_valid
        try:
            fixed = make_valid(geom)
            # If make_valid returns a GeometryCollection, try to extract the largest polygon
            if hasattr(fixed, 'geoms'):
                # Find the largest geometry in the collection
                largest = max(fixed.geoms, key=lambda g: g.area if hasattr(g, 'area') else 0)
                return largest if largest.is_valid else None
            return fixed if fixed.is_valid else None
        except Exception as e:
            print(f"Warning: Could not fix geometry: {e}")
            return None

    return geom

def save_json_result(bbox, country: str):
    """
    Append result to existing JSON file.
    bbox is converted to a list of native Python floats for JSON serialization.
    """

    filename = "calculated_country_bbox.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    logger.info(f"Saving to JSON: {filepath}")

    # Ensure JSON-serializable: native Python floats (numpy scalars are not serializable)
    bbox = [float(x) for x in bbox]

    # check file exists and is valid JSON
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = {}

    data[country] = bbox

    # Sort by country name (alphabetically) so the file stays consistent
    sorted_data = {k: data[k] for k in sorted(data)}

    # save data
    with open(filepath, 'w') as fp:
        json.dump(sorted_data, fp, indent=2)

    logger.info("JSON save complete")


def process(country: str):

    logger.info(f"Processing {country}")

    # get country config
    config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["Other"])
    logger.debug(f"Country config: {config}")

    gdf = load_gdf(config)

    if config["extract_country_required"]:

        gdf = extract_geometry_by_country(gdf)
        save_map(gdf, os.path.join("by_country", f"{country}_map.html"))

        gdf = multipolygon_to_polygons(gdf)
        save_map(gdf, os.path.join("by_country", f"{country}_map_exploded.html"))

    if config["filter_to_main_territory"]:

        gdf = filter_to_main_territory(gdf, "largest_polygons")
        save_map(gdf, os.path.join("by_country", f"{country}_map_exploded_filtered.html"))
    
    # Fix all geometries
    gdf['geometry'] = gdf.geometry.apply(fix_geometry)
    
    gdf = gdf.dissolve()
    # save_map(gdf, os.path.join("by_country", f"{country}_map_dissolved.html"))
    
    gdf = add_buffer(gdf, BUFFER_DISTANCE)
    # Fill any holes (interior rings) in the buffered polygon
    gdf["geometry"] = gdf.geometry.apply(fill_holes)
    save_map(gdf, os.path.join("by_country", f"{country}_map_buffered.html"))

    # GeoPandas total_bounds is (minx, miny, maxx, maxy) = (West, South, East, North)
    bbox = gdf.total_bounds
    # Convert to [North, West, South, East] for CDS API / meteo_data convention
    bbox = [bbox[3], bbox[0], bbox[1], bbox[2]]  # [maxy, minx, miny, maxx]

    save_json_result(bbox, country)

    # save buffered geojson
    geojson_dir = os.path.join(
        OUTPUT_DIR,
        "by_country",
        "buffered_geojsons"
        )
    if not os.path.exists(geojson_dir):
        os.mkdir(geojson_dir)
    filename = f"{country}.geojson"
    filepath = os.path.join(geojson_dir, filename)
    gdf.to_file(filepath, driver='GeoJSON')

    logger.info("bbox processing complete")
    return bbox

def load_gdf(config: dict) -> gpd.GeoDataFrame:

    # load gdf and set up CRS
    filepath = os.path.join(INPUT_DIR, config["filename"])
    gdf = gpd.read_file(filepath, layer=config["layer"])
    if config["crs_epsg"] == 27700:
        # for united kingdom
        gdf = gdf.to_crs(4326)
    else:
        # for geoBoundaries
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    return gdf

if __name__ == "__main__":
    country = "United States"
    bbox = process(country)
    # print(type(bbox))
    logger.info(f"bbox: {bbox}")
