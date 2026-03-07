"""
Extract a single point (lat/lon) timeseries from ERA5 data.

This script reads an ERA5 zip file and extracts all timeseries data
(all variables, all timestamps) for a single geographic point.
"""

import io
import re
import sys
from pathlib import Path
import boto3

import pandas as pd
import xarray as xr
import logging
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

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
MODE = "aws"  # "local" or "aws"
ERA5_PREFIX = "era5"
S3 = boto3.client("s3")

# Hive partition key pattern: key=value (value may contain spaces, e.g. "United Kingdom")
_PARTITION_PATTERN = re.compile(r"^(country|lat|lon)=(.+)$")


def _parse_partition_from_segments(segments: list[str]) -> tuple[str, float, float] | None:
    """Parse Hive-style partition segments into (country, lat, lon). Returns None if invalid."""
    parts = {}
    for seg in segments:
        m = _PARTITION_PATTERN.match(seg.strip())
        if m:
            key, val = m.group(1), m.group(2)
            parts[key] = val
    if set(parts) != {"country", "lat", "lon"}:
        return None
    try:
        return (parts["country"], float(parts["lat"]), float(parts["lon"]))
    except (TypeError, ValueError):
        return None


def list_partitions_from_s3(bucket: str, prefix: str) -> pd.DataFrame:
    """
    List partition keys (country, lat, lon) by listing S3 objects with pagination.
    Returns a DataFrame with columns country, lat, lon.
    """
    logger.info(f"Listing partitions from S3: {bucket}/{prefix}")

    seen: set[tuple[str, float, float]] = set()
    continuation_token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{prefix.rstrip('/')}/", "MaxKeys": 1000}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = S3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            segments = key.split("/")
            # Expect at least era5/country=X/lat=Y/lon=Z/something.parquet
            if len(segments) >= 4:
                parsed = _parse_partition_from_segments(segments[1:4])
                if parsed:
                    seen.add(parsed)
        continuation_token = resp.get("NextContinuationToken")
        if not continuation_token:
            break

    if not seen:
        return pd.DataFrame(columns=["country", "lat", "lon"])
    rows = sorted(seen, key=lambda r: (r[0], r[1], r[2]))
    return pd.DataFrame(rows, columns=["country", "lat", "lon"])


def list_partitions_from_local(base_dir: str) -> pd.DataFrame:
    """
    List partition keys (country, lat, lon) by walking the directory tree.
    Returns a DataFrame with columns country, lat, lon.
    """
    base = Path(base_dir)
    if not base.exists():
        return pd.DataFrame(columns=["country", "lat", "lon"])
    seen: set[tuple[str, float, float]] = set()
    for parquet_path in base.rglob("*.parquet"):
        # path relative to base: e.g. country=X/lat=Y/lon=Z/part-0.parquet
        try:
            rel = parquet_path.relative_to(base)
        except ValueError:
            continue
        segments = [rel.parts[i] for i in range(min(3, len(rel.parts)))]
        if len(segments) < 3:
            continue
        parsed = _parse_partition_from_segments(segments)
        if parsed:
            seen.add(parsed)
    if not seen:
        return pd.DataFrame(columns=["country", "lat", "lon"])
    rows = sorted(seen, key=lambda r: (r[0], r[1], r[2]))
    return pd.DataFrame(rows, columns=["country", "lat", "lon"])


def build_catalogue_table(output_dir: str) -> pd.DataFrame:
    """Build catalogue DataFrame from what exists in S3 or on local disk."""
    if MODE == "aws":
        return list_partitions_from_s3(S3_BUCKET, ERA5_PREFIX)
    return list_partitions_from_local(output_dir)


def get_catalogue_path(output_dir: str) -> str:
    """Return the catalogue file path for current MODE."""
    if MODE == "aws":
        return f"s3://{S3_BUCKET}/era5_catalogue/catalogue.parquet"
    return str(Path(output_dir) / "catalogue.parquet")


def write_catalogue(catalogue_df: pd.DataFrame, output_dir: str) -> None:
    """Overwrite the catalogue Parquet file at the standard location."""
    path = get_catalogue_path(output_dir)
    table = pa.Table.from_pandas(catalogue_df, preserve_index=False)
    if path.startswith("s3://"):
        parts = path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        S3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def read_and_print_catalogue(output_dir: str | None = None) -> pd.DataFrame | None:
    """
    Read the catalogue from the standard path and print it.
    If output_dir is None in local mode, uses the default local output dir.
    Returns the DataFrame for reuse, or None if the file is missing.
    """
    path = get_catalogue_path(output_dir)
    try:
        if path.startswith("s3://"):
            parts = path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            resp = S3.get_object(Bucket=bucket, Key=key)
            buf = io.BytesIO(resp["Body"].read())
            table = pq.read_table(buf)
            df = table.to_pandas()
        else:
            df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("Could not read catalogue at %s: %s", path, e)
        return None
    print(df)
    return df


def save_to_partition_s3(ds: xr.Dataset, country: str, output_dir: str) -> None:
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

    if MODE == "local":
        output_dir = r"C:\Users\Elis\repos\us-datacentres\data\test"
    else:
        output_dir = f"s3://{S3_BUCKET}/era5"

    save_to_partition_s3(ds, country, output_dir)

    catalogue_df = build_catalogue_table(output_dir)
    write_catalogue(catalogue_df, output_dir)
    read_and_print_catalogue(output_dir)

    logger.info("Done")
