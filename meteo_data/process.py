"""
Process ERA5 wind data to calculate mean wind speeds and convert to GeoJSON.

This module handles:
- Reading NetCDF files from ERA5 downloads
- Calculating wind speed from u and v components
- Computing mean wind speeds over time
- Converting point data to polygons (grid or Voronoi)
- Exporting to GeoJSON format for use in Plotly Dash
"""

import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pathlib import Path
from typing import Optional, Union, Literal
import logging
import zipfile
import tempfile
import shutil
from pyogrio.errors import DataSourceError

logger = logging.getLogger(__name__)


def calculate_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculate wind speed from u and v components.
    
    Args:
        u: U component of wind (m/s)
        v: V component of wind (m/s)
        
    Returns:
        Wind speed (m/s)
    """
    return np.sqrt(u**2 + v**2)


def _mask_xarray_to_geometry(ds: xr.Dataset, geometry) -> xr.Dataset:
    """
    Mask an xarray Dataset to a Shapely geometry.
    Grid points outside the geometry are set to NaN; coordinates are unchanged.
    """
    lon_var = ds.coords.get("longitude")
    lat_var = ds.coords.get("latitude")
    if lon_var is None or lat_var is None:
        raise ValueError(
            "Dataset must have 'longitude' and 'latitude' coordinates. "
            f"Available: {list(ds.coords.keys())}"
        )
    lons = np.asarray(lon_var)
    lats = np.asarray(lat_var)
    if lons.ndim == 1 and lats.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lons, lats)
    else:
        lon_2d = lons
        lat_2d = lats
    mask = np.array(
        [
            geometry.contains(Point(float(lon), float(lat)))
            for lon, lat in zip(lon_2d.ravel(), lat_2d.ravel())
        ]
    ).reshape(lon_2d.shape)
    mask_da = xr.DataArray(
        mask,
        coords=[lat_var, lon_var],
        dims=["latitude", "longitude"],
    )
    return ds.where(mask_da)


def clip_era5_zip_to_country_buffer(
    zip_path: Union[str, Path], processed_dir: Path, country: str
) -> Path:
    """
    Clip ERA5 data to the country buffer polygon and save as a new NetCDF file.
    Grid points outside the polygon are set to NaN; the grid extent is unchanged.

    Args:
        zip_path: Path to zip file containing ERA5 NetCDF
        processed_dir: Path to processed directory (contains by_country/...)
        country: Country name (used for buffer GeoJSON filename)

    Returns:
        Path to the saved clipped NetCDF file
    """
    zip_path = Path(zip_path)
    processed_dir = Path(processed_dir)
    
    # Read the country buffer GeoJSON
    country_buffer_path = processed_dir / "by_country" / "buffered_geojsons" / f"{country}.geojson"
    try:
        country_buffer = gpd.read_file(country_buffer_path)
    except DataSourceError as e:
        raise DataSourceError(
            f"Country buffer GeoJSON not found for {country}. "
            f"Run process_geodata/01_get_buffer_geojson_and_bbox.py to generate it. "
            f"Expecting file: {country_buffer_path}"
        ) from e
    
    # Single geometry from gdf (union if multiple rows)
    geometry = unary_union(country_buffer.geometry)

    # Read the ERA5 zip file
    ds = read_era5_netcdf(zip_path)

    # Mask the dataset to the polygon (points outside -> NaN)
    ds = _mask_xarray_to_geometry(ds, geometry)

    # Compute wind speed (abs value) from u and v at each time step, add to dataset, drop u and v
    u_name, v_name = "u100", "v100"
    u = ds[u_name]
    v = ds[v_name]
    wind_speed = calculate_wind_speed(u, v)  # keeps time dimension
    ds = ds.drop_vars([u_name, v_name], errors="ignore")
    ds["wind_speed"] = wind_speed
    # Keep only wind_speed (and all coordinates, including time)
    ds = ds[["wind_speed"]]

    # Save the clipped dataset as a zip containing one .nc file (same format as CDS download)
    out_dir = processed_dir / "by_country" / "era5_clipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / f"{country}_{zip_path.stem}.zip"
    nc_stem = f"{country}_{zip_path.stem}.nc"
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_nc = Path(tmp.name)
    try:
        ds.to_netcdf(tmp_nc)
        with zipfile.ZipFile(clip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_nc, arcname=nc_stem)
    finally:
        tmp_nc.unlink(missing_ok=True)
    logger.info(f"Clipped ERA5 data saved to {clip_path}")
    return clip_path



def read_era5_netcdf(netcdf_path: Union[str, Path]) -> xr.Dataset:
    """
    Read ERA5 NetCDF file.
    
    Args:
        netcdf_path: Path to NetCDF file or zip file containing NetCDF
        
    Returns:
        xarray Dataset with wind data
    """
    netcdf_path = Path(netcdf_path)
    
    # Handle zip files
    if netcdf_path.suffix == ".zip":
        # Extract to a temporary directory that we manage
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(netcdf_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the NetCDF file
            nc_files = list(Path(temp_dir).glob("*.nc"))
            if not nc_files:
                raise ValueError(f"No NetCDF file found in zip: {netcdf_path}")
            if len(nc_files) > 1:
                logger.warning(f"Multiple NetCDF files in zip, using first: {nc_files[0]}")
            
            nc_file_path = nc_files[0]
            
            # Open and fully load the dataset into memory
            # This ensures all data is in memory before we close/delete the temp file
            with xr.open_dataset(nc_file_path) as ds:
                # Load all data into memory
                ds_loaded = ds.load()
                # Make a copy to ensure it's independent of the file
                ds_copy = ds_loaded.copy(deep=True)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return ds_copy
        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    # Read NetCDF directly
    return xr.open_dataset(netcdf_path)


def calculate_mean_wind_speed(
    ds: xr.Dataset,
    time_dim: str = "valid_time"
) -> xr.DataArray:
    """
    Calculate mean wind speed from ERA5 dataset.
    - If the dataset has 'mean_wind_speed', return it (optionally mean over time if present).
    - If the dataset has 'wind_speed' (time, lat, lon), return its mean over time.
    - Otherwise compute from u100 and v100 and mean over time.
    """
    if "mean_wind_speed" in ds.data_vars:
        out = ds["mean_wind_speed"]
        if time_dim in out.dims:
            out = out.mean(dim=time_dim)
        return out
    if "wind_speed" in ds.data_vars:
        return ds["wind_speed"].mean(dim=time_dim)

    u = ds.get("u100")
    v = ds.get("v100")
    if u is None or v is None:
        available_vars = list(ds.data_vars)
        raise ValueError(
            f"Could not find u and v wind components. Available variables: {available_vars}"
        )
    wind_speed = calculate_wind_speed(u, v)
    return wind_speed.mean(dim=time_dim)


def points_to_grid_polygons(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    method: Literal["grid", "voronoi"] = "grid"
) -> gpd.GeoDataFrame:
    """
    Convert point data to polygons (grid cells or Voronoi polygons).
    
    Args:
        lons: Longitude values
        lats: Latitude values
        values: Values to assign to polygons (e.g., mean wind speed)
        method: "grid" for regular grid cells, "voronoi" for Voronoi polygons
        
    Returns:
        GeoDataFrame with polygon geometries and values
    """
    if method == "grid":
        return _points_to_grid_cells(lons, lats, values)
    elif method == "voronoi":
        return _points_to_voronoi(lons, lats, values)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid' or 'voronoi'")


def _points_to_grid_cells(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray
) -> gpd.GeoDataFrame:
    """
    Convert points to regular grid cell polygons.
    
    Args:
        lons: Longitude values
        lats: Latitude values
        values: Values to assign to polygons
        
    Returns:
        GeoDataFrame with grid cell polygons
    """
    # Get unique sorted coordinates
    unique_lons = np.unique(lons)
    unique_lats = np.unique(lats)
    
    # Calculate cell sizes (half the distance between adjacent points)
    if len(unique_lons) > 1:
        lon_spacing = np.diff(unique_lons).mean() / 2
    else:
        lon_spacing = 0.1  # Default spacing
    
    if len(unique_lats) > 1:
        lat_spacing = np.diff(unique_lats).mean() / 2
    else:
        lat_spacing = 0.1  # Default spacing
    
    polygons = []
    polygon_values = []
    
    # Create grid cells for each point
    for lon, lat, value in zip(lons.flatten(), lats.flatten(), values.flatten()):
        # Create square polygon around point
        cell = Polygon([
            (lon - lon_spacing, lat - lat_spacing),
            (lon + lon_spacing, lat - lat_spacing),
            (lon + lon_spacing, lat + lat_spacing),
            (lon - lon_spacing, lat + lat_spacing),
            (lon - lon_spacing, lat - lat_spacing)
        ])
        polygons.append(cell)
        polygon_values.append(float(value))
    
    gdf = gpd.GeoDataFrame(
        {"mean_wind_speed": polygon_values},
        geometry=polygons,
        crs="EPSG:4326"
    )
    
    return gdf


def _points_to_voronoi(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray
) -> gpd.GeoDataFrame:
    """
    Convert points to Voronoi polygons.
    
    Args:
        lons: Longitude values
        lats: Latitude values
        values: Values to assign to polygons
        
    Returns:
        GeoDataFrame with Voronoi polygons
    """
    try:
        from scipy.spatial import Voronoi
    except ImportError:
        raise ImportError(
            "scipy is required for Voronoi polygons. Install with: pip install scipy"
        )
    
    # Flatten coordinates and values
    points = np.column_stack([lons.flatten(), lats.flatten()])
    flat_values = values.flatten()
    
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create polygons from Voronoi regions
    polygons = []
    polygon_values = []
    
    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        # Skip infinite regions
        if -1 in region:
            continue
        
        # Get vertices for this region
        vertices = vor.vertices[region]
        
        # Create polygon
        if len(vertices) >= 3:
            poly = Polygon(vertices)
            polygons.append(poly)
            polygon_values.append(float(flat_values[idx]))
    
    gdf = gpd.GeoDataFrame(
        {"mean_wind_speed": polygon_values},
        geometry=polygons,
        crs="EPSG:4326"
    )
    
    return gdf


def process_era5_to_geojson(
    netcdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    polygon_method: Literal["grid", "voronoi"] = "grid",
    time_dim: str = "valid_time"
) -> gpd.GeoDataFrame:
    """
    Process ERA5 NetCDF file to GeoJSON format.
    
    Args:
        netcdf_path: Path to NetCDF file or zip file
        output_path: Optional path to save GeoJSON file. If None, doesn't save
        polygon_method: Method to convert points to polygons ("grid" or "voronoi")
        time_dim: Name of time dimension in dataset
        
    Returns:
        GeoDataFrame with mean wind speed polygons
    """
    logger.info(f"Reading NetCDF file: {netcdf_path}")
    ds = read_era5_netcdf(netcdf_path)
    
    logger.info("Calculating mean wind speed")
    mean_wind_speed = calculate_mean_wind_speed(ds, time_dim=time_dim)
    
    # Get coordinates
    # ERA5 typically uses 'longitude' and 'latitude'
    lon_var = ds.coords.get("longitude")
    lat_var = ds.coords.get("latitude")
    
    if lon_var is None or lat_var is None:
        available_coords = list(ds.coords.keys())
        raise ValueError(
            f"Could not find longitude/latitude coordinates. Available: {available_coords}"
        )
    
    lons = lon_var.values
    lats = lat_var.values
    
    # Create meshgrid if needed
    if lons.ndim == 1 and lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
    
    logger.info(f"Converting to {polygon_method} polygons")
    gdf = points_to_grid_polygons(
        lons,
        lats,
        mean_wind_speed.values,
        method=polygon_method
    )
    
    if output_path is not None:
        output_path = Path(output_path)
        logger.info(f"Saving to GeoJSON: {output_path}")
        gdf.to_file(output_path, driver="GeoJSON")
        logger.info("GeoJSON saved successfully")
    
    return gdf


def process_era5_zip_to_geojson(
    zip_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    polygon_method: Literal["grid", "voronoi"] = "grid",
    time_dim: str = "valid_time"
) -> gpd.GeoDataFrame:
    """
    Process ERA5 zip file (containing NetCDF) to GeoJSON format.
    
    This is a convenience wrapper around process_era5_to_geojson that handles
    zip file extraction.
    
    Args:
        zip_path: Path to zip file containing NetCDF
        output_path: Optional path to save GeoJSON file
        polygon_method: Method to convert points to polygons ("grid" or "voronoi")
        time_dim: Name of time dimension in dataset
        
    Returns:
        GeoDataFrame with mean wind speed polygons
    """
    return process_era5_to_geojson(
        zip_path,
        output_path=output_path,
        polygon_method=polygon_method,
        time_dim=time_dim
    )
