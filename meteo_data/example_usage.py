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

# Add parent directory to path so we can import meteo_data
# This allows the script to be run from any location
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import logging
from typing import Optional
from meteo_data import (
    download_era5_wind_data,
    process_era5_zip_to_geojson,
    sanity_check_geojson,
)

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
    prefix: str = "mean_wind_speed"
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


def main():
    """Example workflow for processing ERA5 wind data."""

    # Configuration
    # Set month and day to None to download/process entire year
    # Country name must match a key in data/processed/calculated_country_bbox.json
    country = "United Kingdom"
    year = 2025
    month = None  # Set to None for full year, or specify month (1-12)
    day = None    # Set to None for full month/year, or specify day (1-31)

    # Create output directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    # Step 1: Download ERA5 data
    logger.info("Step 1: Downloading ERA5 wind data")
    if month is None and day is None:
        logger.info(f"Downloading full year {year} for {country}")
    elif day is None:
        logger.info(f"Downloading {year}-{month:02d} for {country}")
    else:
        logger.info(f"Downloading {year}-{month:02d}-{day:02d} for {country}")

    try:
        zip_path = download_era5_wind_data(
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

    # Step 2: Process data to GeoJSON
    logger.info("Step 2: Processing data to GeoJSON")
    if month is None and day is None:
        logger.info("Processing full year data - this may take a while and use significant memory...")

    geojson_filename = generate_output_filename(country, year, month, day)
    geojson_path = output_dir / geojson_filename

    try:
        gdf = process_era5_zip_to_geojson(
            zip_path=zip_path,
            output_path=geojson_path,
            polygon_method="grid"  # or "voronoi"
        )
        logger.info(f"Processing complete: {geojson_path}")
        logger.info(f"Number of polygons: {len(gdf)}")
        logger.info(f"Wind speed range: {gdf['mean_wind_speed'].min():.2f} - {gdf['mean_wind_speed'].max():.2f} m/s")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return

    # Step 3: Sanity check plot
    logger.info("Step 3: Creating sanity check plot")

    plot_filename = generate_output_filename(country, year, month, day, prefix="wind_speed_map").replace(".geojson", ".html")
    plot_path = output_dir / plot_filename

    try:
        sanity_check_geojson(
            geojson_path=geojson_path,
            output_plot_path=plot_path,
            backend="plotly"
        )
        logger.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return

    logger.info("=" * 60)
    logger.info("All steps completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
