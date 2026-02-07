"""
Optimisation logic triggered by the Optimise button.
"""
import time
import random
from pathlib import Path
import sys
import logging
import pypsa

# Add repo root to path before importing meteo_data (so script can be run from any location)
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import xarray as xr
from meteo_data.process_era5 import read_era5_netcdf

DATA_DIR = _repo_root / "data"


def pypsa_model():
    n = pypsa.Network()
    n.add("Bus", "zone_1")
    n.add("Bus", "zone_2")
    n.buses



def center_lat_lon(ds: xr.Dataset) -> tuple[float, float]:
    """
    Return (lat, lon) at the centre of the dataset's spatial grid.
    Uses the middle index so the point is an actual grid point.
    """
    lat = ds.coords["latitude"]
    lon = ds.coords["longitude"]
    lat_vals = lat.values
    lon_vals = lon.values
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lat_center = float(lat_vals[len(lat_vals) // 2])
        lon_center = float(lon_vals[len(lon_vals) // 2])
    else:
        lat_center = float((lat_vals.min() + lat_vals.max()) / 2)
        lon_center = float((lon_vals.min() + lon_vals.max()) / 2)
    return lat_center, lon_center


def get_era5_data(country: str, year: int = 2025) -> xr.Dataset:
    """
    Get the ERA5 data for a given country and year (from the clipped zip).
    """
    zip_path = DATA_DIR / "processed" / "by_country" / "era5_clipped" / f"{country}_{year}.zip"
    return read_era5_netcdf(zip_path)


def run_optimisation(**kwargs):
    """
    Placeholder: simulates an optimisation run (1-2 seconds), then returns a result.
    Replace with real logic; kwargs can receive form state (toggles, sliders, tiers).
    """
    duration = random.uniform(1.0, 2.0)
    time.sleep(duration)
    return {
        "status": "ok",
        "message": f"Optimisation complete (simulated {duration:.1f}s).",
        "duration_seconds": round(duration, 2),
    }


def main():
    ds = get_era5_data("United Kingdom", 2025)

    # use a point in the middle of the grid
    lat, lon = center_lat_lon(ds)
    print(lat, lon)

    # extract the data for that coordinate (nearest grid point)
    data = ds.sel(latitude=lat, longitude=lon, method="nearest")
    print(data)


if __name__ == "__main__":
    # main()
    pypsa_model()
