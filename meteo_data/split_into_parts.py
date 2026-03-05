"""
Extract a single point (lat/lon) timeseries from ERA5 data.

This script reads an ERA5 zip file and extracts all timeseries data
(all variables, all timestamps) for a single geographic point.
"""

import sys
import numpy as np
from pathlib import Path
import xarray as xr 
import logging
import pyarrow as pa
import pyarrow.dataset as pa_ds

# Add parent directory to path so we can import meteo_data
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from meteo_data.process_era5 import read_era5_netcdf, save_dataset_to_zip

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

S3_BUCKET = "datacentres-dev-data-207662791637"
MODE = "aws" # "local" or "aws"


def save_to_partition_s3(ds: xr.Dataset, output_dir: Path, country: str) -> None:
    """
    Save an xarray Dataset to a partition in S3.
    """
    df = ds.to_dataframe()
    df = df[["latitude", "longitude", "ssrd", "fdir", "wind_speed_100"]]
    df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
    # round lat and lon to 4 decimal places
    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)
    # add country column
    df["country"] = country
    if MODE == "local":
        output_dir = r"C:\Users\Elis\repos\us-datacentres\data\test"
    else:
        output_dir = f"s3://{S3_BUCKET}/era5"
    pa_ds.write_dataset(
        pa.Table.from_pandas(df),
        base_dir=str(output_dir),
        format="parquet",
        partitioning=pa_ds.partitioning(
            pa.schema([
                ("country", pa.string()),
                ("lat", pa.float32()),
                ("lon", pa.float32())
                ]),
            flavor="hive"
        ),
        existing_data_behavior="overwrite_or_ignore",
    )

def extract_single_point(
    ds: xr.Dataset,
    country: str,
    lat: float,
    lon: float,
    output_path: Path,
) -> xr.Dataset:
    """
    Extract timeseries data for a single lat/lon point from ERA5 zip file.
    
    Args:
        zip_path: Path to ERA5 zip file
        lat: Latitude of point to extract
        lon: Longitude of point to extract
        output_path: Path to save output file
        format: Output format ("netcdf", "parquet", "csv", "json", "excel")
        
    Returns:
        xarray Dataset containing single point timeseries
    """
    logger.info(f"Extracting data for point: ({lat}, {lon})")
    # Select nearest point to requested lat/lon
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')

    save_to_partition_s3(ds_point, "", country)

    return ds_point


def load_data(zip_path: Path) -> xr.Dataset:
    """
    Load data from a directory.
    """
    logger.info(f"Reading data from: {zip_path}")
    ds = read_era5_netcdf(zip_path)

    # Log dataset info
    logger.info(f"Dataset dimensions: {dict(ds.dims)}")
    logger.info(f"Dataset variables: {list(ds.data_vars)}")
    logger.info(f"Latitude range: {float(ds.latitude.min())} to {float(ds.latitude.max())}")
    logger.info(f"Longitude range: {float(ds.longitude.min())} to {float(ds.longitude.max())}")
    
    return ds


def get_valid_coords(ds: xr.Dataset) -> list[tuple[float, float]]:
    """
    Get the valid coordinates from an xarray Dataset.
    """
    # Use the first variable (e.g. ssrd) at the first timestep as a mask
    mask = ds[list(ds.data_vars)[0]].isel(valid_time=0).notnull()
    # Stack lat/lon into a single "points" dimension, then filter
    lat_indices, lon_indices = np.where(mask.values)
    valid_lats = ds.latitude.values[lat_indices]
    valid_lons = ds.longitude.values[lon_indices]
    valid_coords = list(zip(valid_lats, valid_lons))
    return valid_coords


if __name__ == "__main__":
    # Configuration
    data_dir = Path("data")
    input_zip = data_dir / "processed" / "by_country" / "era5_clipped" / "United Kingdom_2025.zip"
    country = "United Kingdom"
    s3_bucket = "us-datacentres-data"
    
    # If the above doesn't exist, try the downloads directory
    if not input_zip.exists():
        input_zip = data_dir / "downloads" / "era5_United Kingdom_2025.zip"
    
    if not input_zip.exists():
        logger.error(f"Input file not found: {input_zip}")
        logger.error("Please check the file path")
        sys.exit(1)
    
    # Choose a point (example: London coordinates)
    latitude = 51.5074  # London latitude
    longitude = -0.1278  # London longitude
    
    # Output file
    output_file = data_dir / "processed" / "single_point_UK_2025.parquet"

    ds = load_data(input_zip)

    valid_coords = get_valid_coords(ds)

    for lat, lon in valid_coords[:4]:
        ds_point = extract_single_point(
            ds=ds,
            country=country,
            lat=lat,
            lon=lon,
            output_path=output_file,
        )
    