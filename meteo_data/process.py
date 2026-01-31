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
from typing import Optional, Union, Literal, List, Dict
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
    wind_speed_heights = [10, 100]
    for height in wind_speed_heights:
        u_name, v_name = f"u{height}", f"v{height}"
        u = ds[u_name]
        v = ds[v_name]
        wind_speed = calculate_wind_speed(u, v)  # keeps time dimension
        ds = ds.drop_vars([u_name, v_name], errors="ignore")
        ds[f"wind_speed_{height}"] = wind_speed

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
            
            # Find all NetCDF files (sorted for deterministic order, e.g. time chunks)
            nc_files = sorted(Path(temp_dir).glob("*.nc"))
            if not nc_files:
                raise ValueError(f"No NetCDF file found in zip: {netcdf_path}")
            if len(nc_files) > 1:
                logger.info(
                    f"Multiple NetCDF files in zip ({len(nc_files)}), merging: "
                    f"{[f.name for f in nc_files]}"
                )

            # Open and fully load each dataset into memory, then combine
            datasets = []
            for nc_file_path in nc_files:
                with xr.open_dataset(nc_file_path) as ds:
                    ds_loaded = ds.load()
                    datasets.append(ds_loaded.copy(deep=True))

            if len(datasets) == 1:
                ds_combined = datasets[0]
            else:
                try:
                    ds_combined = xr.combine_by_coords(
                        datasets,
                        combine_attrs="drop_conflicts",
                        compat="no_conflicts",
                    )
                except ValueError as e:
                    # Fallback: concat along time if all have a time dimension
                    time_dim = None
                    for d in datasets:
                        for c in d.coords:
                            if "time" in c.lower():
                                time_dim = c
                                break
                        if time_dim is not None:
                            break
                    if time_dim is None:
                        time_dim = "time"
                    if all(time_dim in d.dims for d in datasets):
                        ds_combined = xr.concat(datasets, dim=time_dim)
                    else:
                        raise ValueError(
                            f"Cannot merge {len(datasets)} NetCDF files: "
                            f"combine_by_coords failed ({e}); no common time dimension to concat."
                        ) from e

            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

            return ds_combined
        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    # Read NetCDF directly
    return xr.open_dataset(netcdf_path)


def calculate_mean_over_time_values(
    ds: xr.Dataset,
    vals_to_mean: Union[str, List[str]],
    time_dim: str = "valid_time"
) -> xr.Dataset:
    """
    Calculate mean of values from ERA5 dataset over the time dimension,
    then drop the time dimension from the dataset.
    """
    if isinstance(vals_to_mean, str):
        vals_to_mean = [vals_to_mean]
    for val in vals_to_mean:
        if val not in ds.data_vars:
            raise ValueError(f"Value {val} not found in dataset")
        ds[val] = ds[val].mean(dim=time_dim)
    # Drop the time dimension/coordinate so the dataset no longer has valid_time
    if time_dim in ds.dims:
        ds = ds.drop_dims(time_dim)
    return ds


def points_to_grid_polygons(
    lons: np.ndarray,
    lats: np.ndarray,
    values: Union[np.ndarray, Dict[str, np.ndarray]],
    method: Literal["grid", "voronoi"] = "grid",
    lon_spacing: Optional[float] = None,
    lat_spacing: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Convert point data to polygons (grid cells or Voronoi polygons).
    
    Args:
        lons: Longitude values
        lats: Latitude values
        values: Single array (one column) or dict of name -> array (multiple
            columns in output GeoDataFrame)
        method: "grid" for regular grid cells, "voronoi" for Voronoi polygons
        lon_spacing: Optional half-width of grid cells in longitude (so cell
            width = 2*lon_spacing). If provided with lat_spacing, cells tile.
        lat_spacing: Optional half-height of grid cells in latitude.
        
    Returns:
        GeoDataFrame with polygon geometries and value column(s)
    """
    if method == "grid":
        return _points_to_grid_cells(
            lons, lats, values,
            lon_spacing=lon_spacing,
            lat_spacing=lat_spacing,
        )
    elif method == "voronoi":
        return _points_to_voronoi(lons, lats, values)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid' or 'voronoi'")


def _points_to_grid_cells(
    lons: np.ndarray,
    lats: np.ndarray,
    values: Union[np.ndarray, Dict[str, np.ndarray]],
    lon_spacing: Optional[float] = None,
    lat_spacing: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Convert points to regular grid cell polygons.
    
    Args:
        lons: Longitude values
        lats: Latitude values
        values: Single array or dict of name -> array (multiple columns)
        lon_spacing: Optional half-width of each cell (so cells tile). If None,
            inferred from mean spacing of unique lons.
        lat_spacing: Optional half-height of each cell. If None, inferred.
        
    Returns:
        GeoDataFrame with grid cell polygons and value column(s)
    """
    # Cell half-sizes: use explicit spacing if provided, else infer from points
    if lon_spacing is not None and lat_spacing is not None:
        pass  # use provided
    else:
        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        if lon_spacing is None:
            if len(unique_lons) > 1:
                lon_spacing = float(np.diff(unique_lons).mean() / 2)
            else:
                lon_spacing = 0.1
        if lat_spacing is None:
            if len(unique_lats) > 1:
                lat_spacing = float(np.diff(unique_lats).mean() / 2)
            else:
                lat_spacing = 0.1

    is_dict = isinstance(values, dict)
    if is_dict:
        value_arrays = {k: np.asarray(v).flatten() for k, v in values.items()}
        keys = list(value_arrays.keys())
    else:
        value_flat = np.asarray(values).flatten()

    polygons = []
    n_pts = len(lons.flatten())
    if is_dict:
        polygon_values = {k: [] for k in keys}
    else:
        polygon_values_list = []

    for idx in range(n_pts):
        lon = lons.flatten()[idx]
        lat = lats.flatten()[idx]
        cell = Polygon([
            (lon - lon_spacing, lat - lat_spacing),
            (lon + lon_spacing, lat - lat_spacing),
            (lon + lon_spacing, lat + lat_spacing),
            (lon - lon_spacing, lat + lat_spacing),
            (lon - lon_spacing, lat - lat_spacing)
        ])
        polygons.append(cell)
        if is_dict:
            for k in keys:
                polygon_values[k].append(float(value_arrays[k][idx]))
        else:
            polygon_values_list.append(float(value_flat[idx]))

    if is_dict:
        data = dict(polygon_values)
    else:
        data = {"mean_wind_speed": polygon_values_list}
    gdf = gpd.GeoDataFrame(data, geometry=polygons, crs="EPSG:4326")
    return gdf


def _points_to_voronoi(
    lons: np.ndarray,
    lats: np.ndarray,
    values: Union[np.ndarray, Dict[str, np.ndarray]]
) -> gpd.GeoDataFrame:
    """
    Convert points to Voronoi polygons.
    
    Args:
        lons: Longitude values
        lats: Latitude values
        values: Single array or dict of name -> array (multiple columns)
        
    Returns:
        GeoDataFrame with Voronoi polygons and value column(s)
    """
    try:
        from scipy.spatial import Voronoi
    except ImportError:
        raise ImportError(
            "scipy is required for Voronoi polygons. Install with: pip install scipy"
        )

    is_dict = isinstance(values, dict)
    if is_dict:
        flat_values = {k: np.asarray(v).flatten() for k, v in values.items()}
        keys = list(flat_values.keys())
    else:
        flat_values_arr = np.asarray(values).flatten()

    points = np.column_stack([lons.flatten(), lats.flatten()])
    vor = Voronoi(points)

    polygons = []
    if is_dict:
        polygon_values = {k: [] for k in keys}
    else:
        polygon_values_list = []

    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region:
            continue
        vertices = vor.vertices[region]
        if len(vertices) >= 3:
            poly = Polygon(vertices)
            polygons.append(poly)
            if is_dict:
                for k in keys:
                    polygon_values[k].append(float(flat_values[k][idx]))
            else:
                polygon_values_list.append(float(flat_values_arr[idx]))

    if is_dict:
        data = dict(polygon_values)
    else:
        data = {"mean_wind_speed": polygon_values_list}
    gdf = gpd.GeoDataFrame(data, geometry=polygons, crs="EPSG:4326")
    return gdf


def _aggregate_points_to_max_polygons(
    lons: np.ndarray,
    lats: np.ndarray,
    values: Union[np.ndarray, Dict[str, np.ndarray]],
    max_polygons: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Union[np.ndarray, Dict[str, np.ndarray]],
    Optional[float],
    Optional[float],
]:
    """
    Reduce (lons, lats, values) to at most max_polygons points by binning in
    lat/lon and averaging values within each bin. Bins are chosen to preserve
    aspect ratio of the data extent. Returns regular grid centers and cell
    half-sizes so grid polygons tile with no gaps.

    values: Either a single array or a dict of name -> array (all same shape).
    When dict, all arrays are aggregated with the same binning; valid mask
    uses the first array.

    Returns:
        (lons, lats, values, lon_spacing, lat_spacing). When no aggregation
        was done, lon_spacing and lat_spacing are None.
    """
    lons_f = lons.flatten()
    lats_f = lats.flatten()
    if isinstance(values, dict):
        first_key = next(iter(values))
        values_f = values[first_key].flatten()
        is_dict = True
    else:
        values_f = values.flatten()
        is_dict = False
    valid = ~np.isnan(values_f)
    lons_v = lons_f[valid]
    lats_v = lats_f[valid]
    n_points = int(np.sum(valid))
    if n_points <= max_polygons:
        if is_dict:
            out_vals = {k: np.asarray(v).flatten()[valid] for k, v in values.items()}
        else:
            out_vals = np.asarray(values).flatten()[valid]
        return lons_v, lats_v, out_vals, None, None

    extent_lat = float(np.nanmax(lats_f) - np.nanmin(lats_f)) or 1.0
    extent_lon = float(np.nanmax(lons_f) - np.nanmin(lons_f)) or 1.0
    ratio = extent_lat / extent_lon
    n_bins_lon = max(1, int(np.sqrt(max_polygons / ratio)))
    n_bins_lat = max(1, int(np.sqrt(max_polygons * ratio)))
    while n_bins_lat * n_bins_lon > max_polygons:
        if n_bins_lat >= n_bins_lon:
            n_bins_lat -= 1
        else:
            n_bins_lon -= 1
        if n_bins_lat < 1 or n_bins_lon < 1:
            break
    n_bins_lat = max(1, n_bins_lat)
    n_bins_lon = max(1, n_bins_lon)

    lat_min, lat_max = float(np.nanmin(lats_f)), float(np.nanmax(lats_f))
    lon_min, lon_max = float(np.nanmin(lons_f)), float(np.nanmax(lons_f))
    lat_edges = np.linspace(lat_min, lat_max, n_bins_lat + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_bins_lon + 1)

    lon_spacing = (lon_max - lon_min) / (2 * n_bins_lon)
    lat_spacing = (lat_max - lat_min) / (2 * n_bins_lat)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    if is_dict:
        values_v = {k: np.asarray(v).flatten()[valid] for k, v in values.items()}
    else:
        values_v = values_f[valid]

    lons_out = []
    lats_out = []
    if is_dict:
        values_out: Dict[str, list] = {k: [] for k in values_v}
    else:
        values_out_list: list = []

    for i in range(n_bins_lat):
        for j in range(n_bins_lon):
            lat_lo, lat_hi = lat_edges[i], lat_edges[i + 1]
            lon_lo, lon_hi = lon_edges[j], lon_edges[j + 1]
            mask = (
                (lats_v >= lat_lo) & (lats_v <= lat_hi) &
                (lons_v >= lon_lo) & (lons_v <= lon_hi)
            )
            if not np.any(mask):
                continue
            lons_out.append(float(lon_centers[j]))
            lats_out.append(float(lat_centers[i]))
            if is_dict:
                for k in values_out:
                    values_out[k].append(float(np.nanmean(values_v[k][mask])))
            else:
                values_out_list.append(float(np.nanmean(values_v[mask])))

    lons_out_a = np.array(lons_out)
    lats_out_a = np.array(lats_out)
    if is_dict:
        values_out_a = {k: np.array(v) for k, v in values_out.items()}
    else:
        values_out_a = np.array(values_out_list)
    return lons_out_a, lats_out_a, values_out_a, lon_spacing, lat_spacing


def process_era5_to_geojson(
    netcdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    polygon_method: Literal["grid", "voronoi"] = "grid",
    time_dim: str = "valid_time",
    max_polygons: Union[int, None] = None
) -> gpd.GeoDataFrame:
    """
    Process ERA5 NetCDF file to GeoJSON format.

    All data variables in the time-averaged dataset (e.g. wind_speed_100,
    ssrd) are written as properties in the GeoJSON. If max_polygons is set,
    nearby points are binned and values averaged so the output has at most
    that many polygons.

    Args:
        netcdf_path: Path to NetCDF file or zip file
        output_path: Optional path to save GeoJSON file. If None, doesn't save
        polygon_method: Method to convert points to polygons ("grid" or "voronoi")
        time_dim: Name of time dimension in dataset
        max_polygons: If set, aggregate (average) nearby points so output has
            at most this many polygons.

    Returns:
        GeoDataFrame with polygon geometries and one column per data variable
    """
    logger.info(f"Reading NetCDF file: {netcdf_path}")
    ds = read_era5_netcdf(netcdf_path)
    
    logger.info("Calculating mean wind speed")
    ds_time_mean = calculate_mean_over_time_values(
        ds,
        vals_to_mean=["ssrd", "wind_speed_100"],
        time_dim=time_dim
        )
    
    # Get coordinates
    lon_var = ds_time_mean.coords.get("longitude")
    lat_var = ds_time_mean.coords.get("latitude")
    if lon_var is None or lat_var is None:
        raise ValueError(
            f"Could not find longitude/latitude coordinates. "
            f"Available: {list(ds_time_mean.coords.keys())}"
        )

    lons = np.asarray(lon_var.values)
    lats = np.asarray(lat_var.values)
    # All data variables in ds_time_mean go into the GeoJSON
    data_var_names = list(ds_time_mean.data_vars)
    values_dict = {
        name: np.asarray(ds_time_mean[name].values)
        for name in data_var_names
    }

    # Create meshgrid if needed
    if lons.ndim == 1 and lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    grid_lon_spacing = None
    grid_lat_spacing = None
    if max_polygons is not None:
        first_arr = next(iter(values_dict.values()))
        n_points = np.sum(~np.isnan(first_arr))
        if n_points > max_polygons:
            logger.info(
                f"Aggregating {n_points} points to at most {max_polygons} polygons "
                "(regular grid, tiling cells)"
            )
            lons, lats, values_dict, grid_lon_spacing, grid_lat_spacing = _aggregate_points_to_max_polygons(
                lons, lats, values_dict, max_polygons
            )

    logger.info(f"Converting to {polygon_method} polygons (columns: {list(values_dict.keys())})")
    gdf = points_to_grid_polygons(
        lons,
        lats,
        values_dict,
        method=polygon_method,
        lon_spacing=grid_lon_spacing,
        lat_spacing=grid_lat_spacing,
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
    time_dim: str = "valid_time",
    max_polygons: Union[int, None] = None
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
        GeoDataFrame with polygon geometries and one column per data variable
    """
    return process_era5_to_geojson(
        zip_path,
        output_path=output_path,
        polygon_method=polygon_method,
        time_dim=time_dim,
        max_polygons=max_polygons
    )
