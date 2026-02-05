"""
Example usage of the meteo_data module.

This script demonstrates how to:
1. Download ERA5 wind data for a country
2. Process the data to calculate mean wind speeds
3. Convert to GeoJSON format
4. Create a sanity check plot
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr

# Add parent directory to path so we can import meteo_data
# This allows the script to be run from any location
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import logging
from typing import Optional
from meteo_data import (
    download_era5_data,
    process_era5_zip_to_geojson,
    sanity_check_geojson,
)
import meteo_data.process_era5 as md_process

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_output_filename(
    country: str,
    year: int,
    month: Optional[int] = None,
    day: Optional[int] = None,
    prefix: str = "map"
) -> str:
    """
    Generate consistent output filename based on date parameters.

    Args:
        country: Country name
        year: Year
        month: Optional month (1-12)
        day: Optional day (1-31)
        prefix: Filename prefix

    Returns:
        Filename string
    """
    if day is not None:
        return f"{prefix}_{country}_{year}_{month:02d}_{day:02d}.geojson"
    elif month is not None:
        return f"{prefix}_{country}_{year}_{month:02d}.geojson"
    else:
        return f"{prefix}_{country}_{year}.geojson"


def main(
    country: str,
    year: int,
    month: Optional[int] = None,
    day: Optional[int] = None,
    download_only: bool = False
    ) -> Optional[Path]:
    """Workflow for processing ERA5 wind data."""

    # Create output directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Step 1: Download ERA5 data
    logger.info("Step 1: Downloading ERA5 wind data")
    if month is None and day is None:
        logger.info(f"Downloading full year {year} for {country}")
    elif day is None:
        logger.info(f"Downloading {year}-{month:02d} for {country}")
    else:
        logger.info(f"Downloading {year}-{month:02d}-{day:02d} for {country}")#

    try:
        zip_path = download_era5_data(
            country=country,
            year=year,
            month=month,
            day=day,
            output_dir=downloads_dir
        )
        logger.info(f"Download complete: {zip_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("Note: You need to set up CDS API credentials first.")
        logger.error("See: https://cds.climate.copernicus.eu/api-how-to")
        return

    # if download_only:
    #     logger.info("Download completed successfully")
    #     return zip_path

    # Step 2: Clip to country + buffer GeoJSON
    # This takes the data in the ERA5 data from the bounds of a bbox to the bounds of a country + buffer
    logger.info("Step 2: Clipping to country + buffer GeoJSON")
    clip_path = md_process.clip_era5_zip_to_country_buffer(
        zip_path=zip_path,
        processed_dir=processed_dir,
        country=country
    )
    logger.info(f"Clip complete: {clip_path}")

    if download_only:
        logger.info("Download completed successfully")
        return clip_path

    # Step 3: Process data to GeoJSON
    logger.info("Step 3: Processing data to GeoJSON")
    if month is None and day is None:
        logger.info("Processing full year data - this may take a while and use significant memory...")

    geojson_filename = generate_output_filename(country, year, month, day)
    geojson_path = processed_dir / geojson_filename

    try:
        gdf = process_era5_zip_to_geojson(
            zip_path=clip_path,
            output_path=geojson_path,
            polygon_method="grid",  # or "voronoi",
            max_polygons=12000,
            country=country
        )
        logger.info(f"Processing complete: {geojson_path}")
        logger.info(f"Number of polygons: {len(gdf)}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return

    # Step 4: Sanity check plot
    logger.info("Step 4: Creating sanity check plot")

    plot_filename = generate_output_filename(country, year, month, day, prefix="map").replace(".geojson", ".html")
    plot_path = processed_dir / plot_filename

    try:
        sanity_check_geojson(
            geojson_path=geojson_path,
            output_plot_path=plot_path,
            backend="explore",
            # color_on="wind_speed_100"
            color_on="ssrd"
        )
        logger.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return

    logger.info("All steps completed successfully")


if __name__ == "__main__":
    # Configuration
    # Set month and day to None to download/process entire year
    # Country name must match a key in data/processed/calculated_country_bbox.json
    country = "United Kingdom"
    year = 2025
    # month = None  # Set to None for full year, or specify month (1-12)
    # day = None    # Set to None for full month/year, or specify day (1-31)
    # main(country, year, month, day)

    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = None

    all_zip_paths = []
    for month in months:
        try:
            zip_path = main(country, year, month, days, download_only=True)
            if zip_path is not None:
                all_zip_paths.append(zip_path)
        except Exception as e:
            logger.error(f"Error processing {country} {year}-{month:02d}: {e}")
            continue

    # read all downloaded zip files, and combine them into a single dataset
    ds_combined = None
    for zip_path in all_zip_paths:
        ds = md_process.read_era5_netcdf(zip_path)

        # debug - print counts of null and not null values for each variable
        logger.debug(f"Zip file: {zip_path}")
        for var in ds.data_vars:
            logger.debug(f"{var}: {ds[var].isnull().sum().item()} null, {ds[var].notnull().sum().item()} not null")

        if ds_combined is None:
            ds_combined = ds
        else:
            ds_combined = xr.concat([ds_combined, ds], dim="valid_time", join="outer")

    # debug - print counts of null and not null values for each variable
    logger.debug(f"Combined dataset:")
    for var in ds_combined.data_vars:
        logger.debug(f"{var}: {ds_combined[var].isnull().sum().item()} null, {ds_combined[var].notnull().sum().item()} not null")

    # Convert accumulated radiation (J/m²) to flux (W/m²) using time step from valid_time
    J_M2_VARS = ["ssrd", "fdir"]  # ERA5 variables in J/m² (accumulated)
    if "valid_time" in ds_combined.coords:
        diffs = np.diff(ds_combined.valid_time.values)
        if len(diffs) > 0:
            step_seconds = float(np.median(diffs).astype("timedelta64[s]").astype(np.float64))
            if step_seconds > 0:
                for var in J_M2_VARS:
                    if var in ds_combined.data_vars:
                        ds_combined = ds_combined.assign(
                            **{
                                var: (ds_combined[var] / step_seconds).assign_attrs(
                                    units="W m**-2"
                                )
                            }
                        )
    
    data_dir = Path("data")
    zip_path = data_dir / "processed" / "by_country" / "era5_clipped" / f"{country}_{year}.zip"
    logger.info(f"Saving combined dataset to zip: {zip_path}")
    md_process.save_dataset_to_zip(ds_combined, zip_path, f"{country}_{year}.nc")

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    geojson_filename = generate_output_filename(country, year, month=None, day=None)
    geojson_path = processed_dir / geojson_filename

    try:
        gdf = process_era5_zip_to_geojson(
            zip_path=zip_path,
            output_path=geojson_path,
            polygon_method="grid",  # or "voronoi",
            max_polygons=12000,
            country=country
        )
        logger.info(f"Processing complete: {geojson_path}")
        logger.info(f"Number of polygons: {len(gdf)}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")

    logger.info("Creating sanity check plot for combined dataset")

    plot_filename = generate_output_filename(country, year, month=None, day=None, prefix="map").replace(".geojson", ".html")
    plot_path = processed_dir / plot_filename

    try:
        sanity_check_geojson(
            geojson_path=geojson_path,
            output_plot_path=plot_path,
            backend="explore",
            # color_on="wind_speed_100"
            # color_on="ssrd"
            # color_on="fdir"
            color_on="onshore"
        )
        logger.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

    logger.info("All steps completed successfully")
