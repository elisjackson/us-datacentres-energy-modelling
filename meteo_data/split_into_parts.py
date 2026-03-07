"""
Extract a single point (lat/lon) timeseries from ERA5 data.

This script reads an ERA5 zip file and extracts all timeseries data
(all variables, all timestamps) for a single geographic point.
"""

import sys
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

from meteo_data.process_era5 import read_era5_netcdf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

S3_BUCKET = "datacentres-dev-data-207662791637"
MODE = "aws" # "local" or "aws"


def save_to_partition_s3(ds: xr.Dataset, country: str) -> None:
    """
    Save an xarray Dataset to partitioned parquet files, one partition per lat/lon point.

    Converts the full dataset to a DataFrame in one pass, drops land-mask nulls,
    and lets PyArrow handle the partitioning — avoiding a per-point loop.
    """
    logger.info(f"Saving data for: {country}")
    cols = ["ssrd", "fdir", "wind_speed_100"]
    df = ds[cols].to_dataframe().reset_index()
    df = df[["valid_time", "latitude", "longitude"] + cols]
    df.rename(columns={
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }, inplace=True)
    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)
    df["country"] = country
    df.dropna(subset=cols, inplace=True)
    df.sort_values(
        ["country", "lat", "lon", "time"],
        inplace=True
    )
    df = df.reset_index(drop=True)

    if MODE == "local":
        output_dir = r"C:\Users\Elis\repos\us-datacentres\data\test"
    else:
        output_dir = f"s3://{S3_BUCKET}/era5"

    pa_ds.write_dataset(
        pa.Table.from_pandas(df, preserve_index=False),
        base_dir=str(output_dir),
        format="parquet",
        partitioning=pa_ds.partitioning(
            pa.schema([
                ("country", pa.string()),
                ("lat", pa.float32()),
                ("lon", pa.float32()),
            ]),
            flavor="hive",
        ),
        existing_data_behavior="overwrite_or_ignore",
    )


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

    ds = load_data(input_zip)
    save_to_partition_s3(ds, country)
    logger.info("Done")
