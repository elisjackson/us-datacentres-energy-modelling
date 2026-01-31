"""
Meteo data import and processing module.

This module provides functionality to:
- Import wind data from Climate Data Store ERA5 API
- Process NetCDF files to calculate mean wind speeds
- Convert point data to polygons and export to GeoJSON
- Visualize data for sanity checking
"""

from .import_ed5 import (
    get_country_bbox,
    download_era5_wind_data,
    download_era5_wind_data_by_bbox,
    generate_time_list,
    generate_date_list,
)
from .process import (
    calculate_wind_speed,
    read_era5_netcdf,
    calculate_mean_wind_speed,
    points_to_grid_polygons,
    process_era5_to_geojson,
    process_era5_zip_to_geojson,
)
from .visualize import plot_wind_data, sanity_check_geojson

__all__ = [
    # Import functions
    "get_country_bbox",
    "download_era5_wind_data",
    "download_era5_wind_data_by_bbox",
    "generate_time_list",
    "generate_date_list",
    # Process functions
    "calculate_wind_speed",
    "read_era5_netcdf",
    "calculate_mean_wind_speed",
    "points_to_grid_polygons",
    "process_era5_to_geojson",
    "process_era5_zip_to_geojson",
    # Visualization functions
    "plot_wind_data",
    "sanity_check_geojson",
]
