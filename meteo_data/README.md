# Meteo Data Module

This module provides functionality to import and process wind data from the Climate Data Store ERA5 reanalysis dataset.

## Features

- Download ERA5 wind data (u and v components) for any country or custom bounding box
- Process NetCDF files to calculate mean wind speeds
- Convert point data to polygons (grid cells or Voronoi polygons)
- Export to GeoJSON format for use in Plotly Dash applications
- Visualization tools for sanity checking

## Setup

### 1. Install Dependencies

The required dependencies are listed in `pyproject.toml`. Install them with:

```bash
uv sync
```

### 2. CDS API Credentials

To download data from the Climate Data Store, you need to:

1. Create an account at https://cds.climate.copernicus.eu/
2. Get your API key from https://cds.climate.copernicus.eu/api-how-to
3. Create a configuration file at `~/.cdsapirc` with:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API_KEY>
```

Replace `<UID>` and `<API_KEY>` with your actual credentials.

### 3. Country bounding boxes (for country-based download)

Country-based downloads use bboxes from `data/processed/calculated_country_bbox.json`. Ensure that file exists and contains the country you want (keys must match exactly, e.g. `"United Kingdom"`). To generate or update it, run `process_geodata/01_get_bbox.py` for the desired country. For a custom region without using the JSON file, use `download_era5_wind_data_by_bbox` instead.

## Usage

### Basic Workflow

```python
from meteo_data import (
    download_era5_wind_data,
    process_era5_zip_to_geojson,
    sanity_check_geojson,
)

# 1. Download data
# Country name must match a key in data/processed/calculated_country_bbox.json
# (e.g. "United Kingdom", "United States", "Canada")

# For a single day:
zip_path = download_era5_wind_data(
    country="United Kingdom",
    year=2025,
    month=1,
    day=1,
    output_dir="data/downloads"
)

# For a full month (set day=None):
zip_path = download_era5_wind_data(
    country="United Kingdom",
    year=2025,
    month=1,
    day=None,
    output_dir="data/downloads"
)

# For a full year (set month=None and day=None):
zip_path = download_era5_wind_data(
    country="United Kingdom",
    year=2025,
    month=None,
    day=None,
    output_dir="data/downloads"
)

# 2. Process to GeoJSON
gdf = process_era5_zip_to_geojson(
    zip_path=zip_path,
    output_path="data/processed/wind_speed.geojson",
    polygon_method="grid"  # or "voronoi"
)

# 3. Sanity check plot
sanity_check_geojson(
    geojson_path="data/processed/wind_speed.geojson",
    output_plot_path="data/plots/wind_map.html"
)
```

### Example Script

Run the example script from the repo root:

```bash
# Option 1: Run directly (script handles path setup)
uv run python meteo_data/example_usage.py

# Option 2: Run as a module
uv run python -m meteo_data.example_usage
```

Both methods work, and the script can also be run in debug mode from your IDE.

### Custom Bounding Box

If you need data for a region not covered by the country list:

```python
from meteo_data import download_era5_wind_data_by_bbox

# Bounding box: [north, west, south, east]
bbox = [60.0, -10.0, 50.0, 2.0]

zip_path = download_era5_wind_data_by_bbox(
    bbox=bbox,
    year=2025,
    output_dir="data/downloads"
)
```

## Module Structure

- `import_ed5.py`: Functions for downloading ERA5 data from CDS API; country bboxes are read from `data/processed/calculated_country_bbox.json`
- `process.py`: Functions for processing NetCDF files and converting to GeoJSON
- `visualize.py`: Functions for creating sanity check plots
- `example_usage.py`: Example workflow script

## Supported Countries

Country bounding boxes are loaded from `data/processed/calculated_country_bbox.json` (created by the `process_geodata/01_get_bbox.py` script). Use country names that match the keys in that file (e.g. `"United Kingdom"`, `"United States"`, `"Canada"`). To add or update countries, run `process_geodata/01_get_bbox.py` for the desired country and the JSON file will be updated.

## Notes

- ERA5 data is available from 1940 to present (with some delay)
- **Full year downloads**: You can download entire years by setting `month=None` and `day=None`. Full year downloads can be very large (several GB) and may take significant time to download and process
- **Memory usage**: Processing full year data requires significant memory. For large datasets, consider processing by month if you encounter memory issues
- Processing time depends on data size and polygon method (grid is faster than Voronoi)
- GeoJSON output is compatible with Plotly Dash choropleth maps
- The code automatically skips re-downloading if the data file already exists