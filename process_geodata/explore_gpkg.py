import geopandas as gpd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Config ----
GPKG_PATH = r"C:\Users\Elis\repos\us-datacentres\data\downloads\shapefiles\geoBoundariesCGAZ_ADM0.gpkg"
LAYER = "globalADM0"
FILTER_VALUE = None   # set to None for full layer
OUT_HTML_DIR = r"C:\Users\Elis\repos\us-datacentres\data\processed"
OUT_HTML_FILENAME = "globalADM0_map.html"

def save_map(gdf: gpd.GeoDataFrame, filename: str):

    m = gdf.explore(
        column="shapeName" if "shapeName" in gdf.columns else None,
        tooltip=True,
        popup=True,
        tiles="CartoDB positron"
    )
    filepath = os.path.join(OUT_HTML_DIR, filename)
    m.save(filepath)
    logger.info(f"Map saved to {filepath}")

# ---- Load data (only when run as script, not when imported) ----
if __name__ == "__main__":
    if FILTER_VALUE:
        gdf = gpd.read_file(
            GPKG_PATH,
            layer=LAYER,
            where=f"shapeName = '{FILTER_VALUE}'"
        )
    else:
        gdf = gpd.read_file(GPKG_PATH, layer=LAYER)

    # Ensure WGS84 for web maps
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    save_map(gdf, OUT_HTML_FILENAME)
