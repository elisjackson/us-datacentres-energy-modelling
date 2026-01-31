"""
Import wind data from Climate Data Store ERA5 API.

This module handles downloading wind data (u and v components) from the
Copernicus Climate Data Store ERA5 reanalysis dataset.
"""

import cdsapi
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


# Country bounding boxes (North, West, South, East)
# Format: [north, west, south, east] in decimal degrees
COUNTRY_BBOX_FILE = r"C:\Users\Elis\repos\us-datacentres\data\processed\calculated_country_bbox.json"


def get_country_bbox(country: str) -> List[float]:
    """
    Get bounding box for a given country.
    
    Args:
        country: Country name (e.g., "UK", "United Kingdom", "US")
        
    Returns:
        Bounding box as [north, west, south, east] in decimal degrees
        
    Raises:
        ValueError: If country is not found in the mapping
    """
    with open(COUNTRY_BBOX_FILE, 'r') as file:
        data = json.load(file)
    bbox = data.get(country)
    if bbox is None:
        countries = data.keys()
        logger.error(f"Country not found. Available countries: {countries}")
        raise ValueError
    return bbox


def generate_time_list(start_hour: int = 0, end_hour: int = 23) -> List[str]:
    """
    Generate list of time strings for API request.
    
    Args:
        start_hour: Starting hour (0-23)
        end_hour: Ending hour (0-23), inclusive
        
    Returns:
        List of time strings in format "HH:MM"
    """
    return [f"{hour:02d}:00" for hour in range(start_hour, end_hour + 1)]


def generate_date_list(year: int, month: Optional[int] = None, 
                       day: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate date lists for API request.
    
    Args:
        year: Year (e.g., 2025)
        month: Month (1-12), if None, all months are included
        day: Day (1-31), if None, all days in month are included
        
    Returns:
        Tuple of (years, months, days) as lists of strings
    """
    years = [str(year)]
    
    if month is None:
        months = [f"{m:02d}" for m in range(1, 13)]
    else:
        months = [f"{month:02d}"]
    
    if day is None:
        # Get all days in the month(s)
        if month is None:
            # All months - use 31 days (API will handle invalid dates)
            days = [f"{d:02d}" for d in range(1, 32)]
        else:
            # Specific month - calculate actual days
            import calendar
            days_in_month = calendar.monthrange(year, month)[1]
            days = [f"{d:02d}" for d in range(1, days_in_month + 1)]
    else:
        days = [f"{day:02d}"]
    
    return years, months, days


def download_era5_wind_data(
    country: str,
    year: int,
    month: Optional[int] = None,
    day: Optional[int] = None,
    output_dir: Optional[Path] = None,
    client: Optional[cdsapi.Client] = None
) -> Path:
    """
    Download ERA5 wind data for a given country and time period.
    
    Args:
        country: Country name (e.g., "UK", "US")
        year: Year to download data for
        month: Optional month (1-12), if None downloads all months
        day: Optional day (1-31), if None downloads all days
        output_dir: Directory to save downloaded files. If None, uses current directory
        client: Optional cdsapi.Client instance. If None, creates a new one
        
    Returns:
        Path to the downloaded file
        
    Raises:
        ValueError: If country is not found
        RuntimeError: If download fails
    """
    # Get bounding box for country
    bbox = get_country_bbox(country)
    
    # Generate date lists
    years, months, days = generate_date_list(year, month, day)
    
    # Generate time list (all 24 hours)
    times = generate_time_list(0, 23)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if day is not None:
        filename = f"era5_wind_{country}_{year}_{month:02d}_{day:02d}.zip"
    elif month is not None:
        filename = f"era5_wind_{country}_{year}_{month:02d}.zip"
    else:
        filename = f"era5_wind_{country}_{year}.zip"
    
    output_path = output_dir / filename
    
    # Check if file already exists
    if output_path.exists():
        logger.info(f"ERA5 data file already exists: {output_path}")
        logger.info("Skipping download and using existing file.")
        return output_path
    
    # Create API request
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "year": years,
        "month": months,
        "day": days,
        "time": times,
        "data_format": "netcdf",
        "download_format": "zip",
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind"
        ],
        "area": bbox  # [North, West, South, East]
    }
    
    logger.info(f"Downloading ERA5 data for {country}, year {year}")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Output file: {output_path}")
    
    # Create client if not provided
    if client is None:
        client = cdsapi.Client()
    
    try:
        # Download data
        client.retrieve(dataset, request).download(str(output_path))
        logger.info(f"Download complete: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(f"Failed to download ERA5 data: {e}") from e


def download_era5_wind_data_by_bbox(
    bbox: List[float],
    year: int,
    month: Optional[int] = None,
    day: Optional[int] = None,
    output_dir: Optional[Path] = None,
    client: Optional[cdsapi.Client] = None,
    filename_prefix: str = "era5_wind"
) -> Path:
    """
    Download ERA5 wind data for a custom bounding box.
    
    Args:
        bbox: Bounding box as [north, west, south, east] in decimal degrees
        year: Year to download data for
        month: Optional month (1-12), if None downloads all months
        day: Optional day (1-31), if None downloads all days
        output_dir: Directory to save downloaded files. If None, uses current directory
        client: Optional cdsapi.Client instance. If None, creates a new one
        filename_prefix: Prefix for output filename
        
    Returns:
        Path to the downloaded file
    """
    # Generate date lists
    years, months, days = generate_date_list(year, month, day)
    
    # Generate time list (all 24 hours)
    times = generate_time_list(0, 23)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if day is not None:
        filename = f"{filename_prefix}_{year}_{month:02d}_{day:02d}.zip"
    elif month is not None:
        filename = f"{filename_prefix}_{year}_{month:02d}.zip"
    else:
        filename = f"{filename_prefix}_{year}.zip"
    
    output_path = output_dir / filename
    
    # Check if file already exists
    if output_path.exists():
        logger.info(f"ERA5 data file already exists: {output_path}")
        logger.info("Skipping download and using existing file.")
        return output_path
    
    # Create API request
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "year": years,
        "month": months,
        "day": days,
        "time": times,
        "data_format": "netcdf",
        "download_format": "zip",
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind"
        ],
        "area": bbox  # [North, West, South, East]
    }
    
    logger.info(f"Downloading ERA5 data for bbox {bbox}, year {year}")
    logger.info(f"Output file: {output_path}")
    
    # Create client if not provided
    if client is None:
        client = cdsapi.Client()
    
    try:
        # Download data
        client.retrieve(dataset, request).download(str(output_path))
        logger.info(f"Download complete: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(f"Failed to download ERA5 data: {e}") from e
