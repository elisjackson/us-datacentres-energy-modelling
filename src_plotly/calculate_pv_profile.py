"""
Example: Using ERA5 solar radiation data with pvlib to generate hourly kW profiles

This script demonstrates how to:
1. Load and process ERA5 ssrd and fdir data
2. Convert to irradiance components (GHI, DNI, DHI)
3. Use pvlib to model a PV system and generate power output

Alternatively, use pvlib's built-in weather data sources (PVGIS) for comparison
"""

import pandas as pd
import numpy as np
from pvlib import location, pvsystem, modelchain, temperature, iotools
from pathlib import Path

def prepare_era5_data(ssrd, fdir, timestamps):
    """
    Convert ERA5 accumulated radiation to instantaneous irradiance (W/m²)
    
    Parameters:
    -----------
    ssrd : array-like
        Surface solar irradiance downwards (W/m²)
    fdir : array-like
        Total sky direct solar irradiance at surface (W/m²)
    timestamps : DatetimeIndex
        Hourly timestamps
    
    Returns:
    --------
    DataFrame with GHI, DNI, DHI in W/m²
    """
    
    ghi = ssrd
    dni_times_cos = fdir
    
    # Set negative values to zero (nighttime)
    ghi = np.maximum(ghi, 0)
    dni_times_cos = np.maximum(dni_times_cos, 0)
    
    # Calculate diffuse horizontal irradiance
    dhi = ghi - dni_times_cos
    dhi = np.maximum(dhi, 0)  # Ensure non-negative
    
    # Create DataFrame
    irradiance = pd.DataFrame({
        'ghi': ghi,
        'dni': np.nan,  # This will be calculated later
        'dhi': dhi
    }, index=timestamps)
    
    return irradiance


def calculate_dni_from_fdir(fdir, irradiance_df, solar_position):
    """
    Calculate DNI directly from fdir (direct radiation on horizontal surface)
    
    Since fdir = DNI x cos(zenith), then:
    DNI = fdir / cos(zenith)
    """
    
    cos_zenith = np.cos(np.radians(solar_position['apparent_zenith']))
    
    # Avoid division by zero/very small numbers when sun is near horizon
    cos_zenith = np.maximum(cos_zenith, 0.01)
    
    # Calculate DNI directly from the direct horizontal component
    dni = fdir / cos_zenith
    
    # Set to zero when sun is below / near horizon
    dni[solar_position['apparent_zenith'] > 87] = 0
    dni = np.maximum(dni, 0)
    
    irradiance_df['dni'] = dni
    
    return irradiance_df


def get_pvgis_weather(lat: float, lon: float):
    """
    Fetch TMY (Typical Meteorological Year) weather data from PVGIS
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    
    Returns:
    --------
    DataFrame with GHI, DNI, DHI, temp_air, wind_speed
    """
    print(f"\nFetching PVGIS TMY data for lat={lat}, lon={lon}...")
    
    # Get TMY data from PVGIS - returns (data, inputs, metadata)
    # Using map_variables=True to get standard pvlib column names
    result = iotools.get_pvgis_tmy(lat, lon, map_variables=True)
    
    # Handle different return formats
    if len(result) == 3:
        weather, inputs, metadata = result
    elif len(result) == 2:
        weather, metadata = result
        inputs = {}
    else:
        raise ValueError(f"Unexpected return format from get_pvgis_tmy: {len(result)} values")
    
    print(f"PVGIS data retrieved: {len(weather)} hours")
    print(f"Available columns: {list(weather.columns)}")
    
    return weather


def main(lat: float, lon: float, data_source: str = 'era5'):
    """
    Run PV simulation with specified weather data source
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    data_source : str, default 'era5'
        Weather data source: 'era5' or 'pvgis'
    """
    
    altitude = 1  # meters
    timezone = 'UTC'
    
    # Create location object
    site = location.Location(lat, lon, tz=timezone, altitude=altitude)
    
    # =============================================================================
    # STEP 2: Load weather data based on source
    # =============================================================================
    
    if data_source.lower() == 'pvgis':
        # Use PVGIS data
        weather_data = get_pvgis_weather(lat, lon)
        timestamps_local = weather_data.index
        
        # PVGIS data already has GHI, DNI, DHI
        irradiance = pd.DataFrame({
            'ghi': weather_data['ghi'],
            'dni': weather_data['dni'],
            'dhi': weather_data['dhi'],
        }, index=timestamps_local)
        
        print(f"\nPVGIS Irradiance statistics (W/m²):")
        print(irradiance.describe())
        
    elif data_source.lower() == 'era5':
        # Load ERA5 single-point timeseries data
        data_path = Path("data/processed/single_point_UK_2025.parquet")

        if not data_path.exists():
            raise FileNotFoundError(
                f"ERA5 data file not found: {data_path}\n"
                "Run meteo_data/split_into_parts.py first to generate the single-point data."
            )

        # Read the parquet file
        era5_df = pd.read_parquet(data_path)

        # Extract radiation variables (W/m²)
        ssrd = era5_df['ssrd'].values  # Surface solar radiation downwards
        fdir = era5_df['fdir'].values  # Total sky direct solar radiation at surface
        
        # Diagnostic: check raw ERA5 values
        print(f"\nRaw ERA5 data statistics:")
        print(f"  ssrd - mean: {ssrd.mean():.1f}, max: {ssrd.max():.1f}, min: {ssrd.min():.1f}")
        print(f"  fdir - mean: {fdir.mean():.1f}, max: {fdir.max():.1f}, min: {fdir.min():.1f}")
        print(f"  Expected mean GHI for UK: ~50-100 W/m²")
        print(f"  Expected max GHI for UK: ~800-1000 W/m²")

        # Get timestamps from the index
        timestamps = era5_df.index
        timestamps_local = timestamps
        
        # Calculate solar position for ERA5 processing
        print("\nCalculating solar position...")
        solar_position = site.get_solarposition(timestamps_local)
        
        # Process ERA5 data to irradiance components
        print("Processing ERA5 data to irradiance components...")
        irradiance = prepare_era5_data(ssrd, fdir, timestamps_local)
        irradiance = calculate_dni_from_fdir(fdir, irradiance, solar_position)

        print(f"\nERA5 Irradiance statistics (W/m²):")
        print(irradiance.describe())
        
    else:
        raise ValueError(f"Unknown data source: {data_source}. Use 'era5' or 'pvgis'")

    # =============================================================================
    # STEP 6: Define PV system parameters
    # =============================================================================

    dc_rating = 1000  # W

    # Module parameters (example: Canadian Solar CS6K-280M)
    module_parameters = {
        'pdc0': dc_rating,  # DC power at STC (W)
        'gamma_pdc': -0.0045,  # Temperature coefficient (%/°C)
    }

    dc_ac_ratio = 1.25
    ac_rating = dc_rating / dc_ac_ratio

    # Alternative: Use CEC module database
    # modules = pvlib.pvsystem.retrieve_sam('CECMod')
    # module = modules['Canadian_Solar_CS6K_280M']

    # Inverter parameters (example: ABB PVS-100-TL)
    inverter_parameters = {
        'pdc0': ac_rating,
        'eta_inv_nom': 0.96,  # Nominal inverter efficiency
    }

    # Alternative: Use CEC inverter database
    # inverters = pvlib.pvsystem.retrieve_sam('CECInverter')
    # inverter = inverters['ABB__PVS_100_TL__480V_']

    # System configuration
    system_parameters = {
        'surface_tilt': 35,  # Panel tilt angle (degrees)
        'surface_azimuth': 180,  # Panel azimuth (180 = south in Northern hemisphere)
        'modules_per_string': 1,
        'strings_per_inverter': 1,
    }

    # Calculate total system DC capacity
    total_modules = system_parameters['modules_per_string'] * system_parameters['strings_per_inverter']
    system_dc_capacity = total_modules * module_parameters['pdc0'] / 1000  # kW

    print(f"\nPV System Configuration:")
    print(f"  DC capacity: {system_dc_capacity:.1f} kW")

    # =============================================================================
    # STEP 7: Create PV system and model chain
    # =============================================================================

    # Temperature model parameters
    temp_model_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # Create PVSystem
    system = pvsystem.PVSystem(
        surface_tilt=system_parameters['surface_tilt'],
        surface_azimuth=system_parameters['surface_azimuth'],
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters,
        modules_per_string=system_parameters['modules_per_string'],
        strings_per_inverter=system_parameters['strings_per_inverter'],
        temperature_model_parameters=temp_model_params
    )

    # Create ModelChain
    mc = modelchain.ModelChain(
        system,
        site,
        aoi_model='physical',
        spectral_model='no_loss',
    )

    # =============================================================================
    # STEP 8: Run the model to get power output
    # =============================================================================

    print("\nRunning PV model...")

    weather = pd.DataFrame({
        'ghi': irradiance['ghi'],
        'dhi': irradiance['dhi'],
        'dni': irradiance['dni'],
    }, index=timestamps_local)

    # Run the model
    mc.run_model(weather)

    # Extract results
    ac_power = mc.results.ac  # AC power in watts
    dc_power = mc.results.dc  # DC power (array output)

    # Convert to kW
    ac_power_kw = ac_power / 1000
    dc_power_kw = dc_power[0] / 1000  # dc is a tuple, take first element

    # Create results DataFrame
    results = pd.DataFrame({
        'ac_power_kw': ac_power_kw,
        'dc_power_kw': dc_power_kw,
        'ghi': weather['ghi'],
        'dni': weather['dni'],
        'dhi': weather['dhi'],
    }, index=timestamps_local)

    print("\nPower output statistics (kW):")
    print(results[['ac_power_kw', 'dc_power_kw']].describe())

    # Calculate annual energy
    annual_energy_kwh = results['ac_power_kw'].sum()
    print(f"\nAnnual AC energy production: {annual_energy_kwh:,.0f} kWh")
    print(f"Capacity factor: {1e3 * annual_energy_kwh / (ac_rating * 8760) * 100:.1f}%")

    return results


if __name__ == "__main__":
    # UK coordinates (approximately Reading)
    lat, lon = 52.5, -0.75
    
    # Choose data source: 'era5' or 'pvgis'
    # Use 'pvgis' to validate results with pvlib's built-in data
    data_source = 'era5'  # Change to 'pvgis' to compare
    
    print(f"\n{'='*60}")
    print(f"Running PV simulation with {data_source.upper()} data")
    print(f"Location: lat={lat}, lon={lon}")
    print(f"{'='*60}")
    
    results = main(lat, lon, data_source=data_source)
    
    # Optional: Run both sources for comparison
    # print("\n" + "="*60)
    # print("COMPARISON MODE - Running both data sources")
    # print("="*60)
    # results_era5 = main(lat, lon, data_source='era5')
    # results_pvgis = main(lat, lon, data_source='pvgis')
    # print("\nComparison of annual energy production:")
    # print(f"ERA5:  {results_era5['ac_power_kw'].sum():,.0f} kWh")
    # print(f"PVGIS: {results_pvgis['ac_power_kw'].sum():,.0f} kWh")
