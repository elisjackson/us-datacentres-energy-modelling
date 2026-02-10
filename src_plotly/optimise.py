"""
Optimisation logic triggered by the Optimise button.
"""
import time
import random
from pathlib import Path
import sys
import logging
import pandas as pd
import numpy as np
import pypsa

# Add repo root to path before importing meteo_data (so script can be run from any location)
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import xarray as xr
from meteo_data.process_era5 import read_era5_netcdf

DATA_DIR = _repo_root / "data"

logger = logging.getLogger(__name__)


def execute_optimisation(
    **kwargs: dict,
    ):
    """
    Execute the optimisation.

    Args:
        kwargs: dict, containing the following keys:
            "load": float, data centre load in MW
            "wind_cost_sel": str, "Low", "Mid", "High"
            "pv_cost_sel": str, "Low", "Mid", "High"
            "ccgt_capex_sel": str, "Low", "Mid", "High"
            "ccgt_marginal_cost_sel": str, "Low", "Mid", "High"

    kwargs: {
        'toggles': {
            'Data Centre Capacity': True,
            'Wind': True,
            'Solar PV': True,
            'Grid Connection': True,
            'Gas CCGT': True,
            'SMR': True,
            'CO2 price': True
            },
        'sliders': {
            'Data Centre Capacity': 50,
            'Wind': 50,
            'Solar PV': 50,
            'Grid Connection': 50,
            'Gas CCGT': 50,
            'SMR': 50,
            'CO2 price': 50,
        },
        'tiers': {
            'Data Centre Capacity': 'Mid',
            'Wind': 'Mid',
            'Solar PV': 'Mid',
            'Grid Connection': 'Mid',
            'Gas CCGT': 'Mid',
            'SMR': 'Mid',
            'CO2 price': 'Mid'
    """

    logger.info(f"kwargs: {kwargs}")
    print(f"kwargs: {kwargs}")

    t0 = time.time()

    costs = get_technology_costs(
        wind_sel=kwargs["tiers"]["Wind"],
        pv_sel=kwargs["tiers"]["Solar PV"],
        ccgt_capex_sel=kwargs["tiers"]["Gas CCGT"],
        # ccgt_marginal_cost_sel=kwargs["tiers"]["CCGT Marginal Cost"],
        )
    
    network = pypsa_model_2(
        load=kwargs["sliders"]["Data Centre Capacity"],
        wind_cost=costs["wind"],
        solar_cost=costs["pv"],
        ccgt_cost=costs["ccgt_capex"],
        ccgt_marginal_cost=costs["ccgt_marginal_cost"],
    )

    t1 = time.time()
    duration = t1 - t0

    # things we want to return:
    # System costs (CAPEX, OPEX, Energy cost, CO2 Total)
    # Total CO2 per year
    # Generation and Storage optimal capacities
    # Generation and Storage CAPEX, OPEX, Energy costs, CO2
    # Generation and Storage optimal dispatch (timeseries)

    total_cost = network.objective

    # total energy demand
    network.loads_t.p.sum().to_dict()

    # total energy delivered per generator
    total_energy = network.generators_t.p.sum().to_dict()

    # Calculate emissions using carrier information
    total_emissions = network.generators_t.p.multiply(
        network.generators.carrier.map(network.carriers.co2_emissions)
    ).sum().sum()
    
    # timeseries of generation
    generation_ts_dict = {}
    generation_ts = network.generators_t.p
    generation_ts.index = range(0, len(generation_ts))
    # generation_ts.to_dict()
    for gen in generation_ts.columns:
        generation_ts_dict[gen] = generation_ts["wind"].to_list()

    # optimal generation capacities
    generator_stats = network.generators[["p_nom_opt", "capital_cost", "lifetime", "marginal_cost"]].to_dict()

    # optimal storage capacities
    storage_stats = network.storage_units[["p_nom_opt", "capital_cost", "lifetime", "marginal_cost"]].to_dict()

    return {
        "status": "ok",
        "total_cost": total_cost,
        "total_energy": total_energy,
        "total_emissions": total_emissions,
        "generation_ts": generation_ts_dict,
        "generator_stats": generator_stats,
        "storage_stats": storage_stats,
        "duration_seconds": round(duration, 2),
    }

def get_technology_costs(
    wind_sel: str,
    pv_sel: str,
    ccgt_capex_sel: str,
    ccgt_marginal_cost_sel: str = "Mid",
    ):
    """
    Get the technology costs for the selected wind and pv options.

    Args:
        wind_sel: str, "Low", "Mid", "High"
        pv_sel: str, "Low", "Mid", "High"

    Returns:
        dict, {"wind": float, "pv": float}
    """

    costs = {
        "wind": {
            "Low": 5000,
            "Mid": 10000,
            "High": 15000,
        },
        "pv": {
            "Low": 5000,
            "Mid": 10000,
            "High": 15000,
        },
        "ccgt_capex": {
            "Low": 5000,
            "Mid": 1000,
            "High": 1500,
        },
        "ccgt_marginal_cost": {
            "Low": 50,
            "Mid": 100,
            "High": 150,
        },
    }

    costs_sel = {
        "wind": costs["wind"][wind_sel],
        "pv": costs["pv"][pv_sel],
        "ccgt_capex": costs["ccgt_capex"][ccgt_capex_sel],
        "ccgt_marginal_cost": costs["ccgt_marginal_cost"][ccgt_marginal_cost_sel],
    }

    return costs_sel


def pypsa_model_2(
    load: float,
    wind_cost: float,
    solar_cost: float,
    ccgt_cost: float,
    ccgt_marginal_cost: float,
) -> pypsa.Network:

    logger.info("Creating PyPSA network")
    # Create network with snapshots
    network = pypsa.Network()
    snapshots = pd.date_range("2024-01-01", periods=8760, freq="h")  # Full year
    network.set_snapshots(snapshots)

    # Add carrier for gas
    network.add("Carrier", "gas", co2_emissions=0.35)  # tonnes/MWh for CCGT

    # CO2 price
    co2_price = 80  # €/tonne

    # Add bus
    network.add("Bus", "bus_0")

    # Add constant load
    network.add("Load",
                "load",
                bus="bus_0",
                p_set=load)

    # Create realistic wind profile for a year
    np.random.seed(42)
    hours = np.arange(8760)
    wind_profile = np.clip(
        0.35 +  # base capacity factor
        0.15 * np.sin(2 * np.pi * hours / 8760) +  # seasonal variation
        0.20 * np.sin(2 * np.pi * hours / 24) +    # diurnal variation
        0.15 * np.random.randn(8760),              # randomness
        0, 1
    )

    # Create realistic solar profile for a year
    np.random.seed(42)
    hours = np.arange(8760)
    solar_profile = np.clip(
        0.15 +  # base capacity factor
        0.15 * np.sin(2 * np.pi * hours / 8760) +  # seasonal variation
        0.20 * np.sin(2 * np.pi * hours / 24) +    # diurnal variation
        0.15 * np.random.randn(8760),              # randomness
        0, 1
    )

    # Add wind farm - capacity is OPTIMIZABLE
    network.add("Generator",
                "wind",
                bus="bus_0",
                p_nom_extendable=True,  # KEY: make capacity a decision variable
                p_nom_min=0,            # minimum capacity
                p_nom_max=500,          # maximum capacity (optional constraint)
                capital_cost=wind_cost,   # cost per MW per year (annualized CAPEX)
                marginal_cost=0,        # €0/MWh operational cost
                p_max_pu=wind_profile)

    # Add solar farm - capacity is OPTIMIZABLE
    network.add("Generator",
                "solar",
                bus="bus_0",
                p_nom_extendable=True,  # KEY: make capacity a decision variable
                p_nom_min=0,            # minimum capacity
                p_nom_max=500,          # maximum capacity (optional constraint)
                capital_cost=solar_cost,   # cost per MW per year (annualized CAPEX)
                marginal_cost=0,        # €0/MWh operational cost
                p_max_pu=solar_profile)

    # Gas generator - add CO2 price to marginal cost
    gas_fuel_cost = 50
    gas_co2_cost = network.carriers.co2_emissions['gas'] * co2_price

    # Add CCGT - capacity is OPTIMIZABLE
    network.add("Generator",
                "ccgt",
                bus="bus_0",
                carrier="gas",
                p_nom_extendable=True,  # KEY: make capacity a decision variable
                p_nom_min=0,
                p_nom_max=300,
                capital_cost=ccgt_cost,    # cost per MW per year (annualized CAPEX)
                marginal_cost=gas_fuel_cost + gas_co2_cost)       # operational cost per MWh

    # Add Storage
    network.add("StorageUnit",
                "storage",
                bus="bus_0",
                p_nom_extendable=True,  # KEY: make capacity a decision variable
                max_hours=4,
                capital_cost=1000,
                marginal_cost=0)

    # Solve for optimal capacities AND dispatch
    logger.info("Solving optimization...")
    t0 = time.time()
    network.optimize()
    t1 = time.time()
    logger.info(f"Optimization completed in {t1 - t0:.2f} seconds")

    # Results
    print("=== OPTIMAL CAPACITIES ===")
    print(f"Wind capacity: {network.generators.p_nom_opt['wind']:.2f} MW")
    print(f"Solar capacity: {network.generators.p_nom_opt['solar']:.2f} MW")
    print(f"CCGT capacity: {network.generators.p_nom_opt['ccgt']:.2f} MW")

    print("\n=== COSTS ===")
    print(f"Total system cost: €{network.objective:,.2f}/year")

    # Break down costs
    wind_capex = network.generators.p_nom_opt['wind'] * network.generators.capital_cost['wind']
    solar_capex = network.generators.p_nom_opt['solar'] * network.generators.capital_cost['solar']
    ccgt_capex = network.generators.p_nom_opt['ccgt'] * network.generators.capital_cost['ccgt']
    wind_opex = (network.generators_t.p['wind'] * network.generators.marginal_cost['wind']).sum()
    solar_opex = (network.generators_t.p['solar'] * network.generators.marginal_cost['solar']).sum()
    ccgt_opex = (network.generators_t.p['ccgt'] * network.generators.marginal_cost['ccgt']).sum()

    print(f"\nWind CAPEX: €{wind_capex:,.2f}/year")
    print(f"Wind OPEX:  €{wind_opex:,.2f}/year")
    print(f"Solar CAPEX: €{solar_capex:,.2f}/year")
    print(f"Solar OPEX:  €{solar_opex:,.2f}/year")
    print(f"CCGT CAPEX: €{ccgt_capex:,.2f}/year")
    print(f"CCGT OPEX:  €{ccgt_opex:,.2f}/year")

    print("\n=== ENERGY DELIVERED ===")
    wind_energy = network.generators_t.p['wind'].sum()
    solar_energy = network.generators_t.p['solar'].sum()
    ccgt_energy = network.generators_t.p['ccgt'].sum()
    total_demand = 100 * 8760  # MW * hours

    print(f"Wind: {wind_energy:,.0f} MWh ({wind_energy/total_demand:.1%} of demand)")
    print(f"Solar: {solar_energy:,.0f} MWh ({solar_energy/total_demand:.1%} of demand)")
    print(f"CCGT: {ccgt_energy:,.0f} MWh ({ccgt_energy/total_demand:.1%} of demand)")

    print("\n=== CAPACITY FACTORS ===")
    print(f"Wind CF: {wind_energy / (network.generators.p_nom_opt['wind'] * 8760):.1%}")
    print(f"Solar CF: {solar_energy / (network.generators.p_nom_opt['solar'] * 8760):.1%}")
    print(f"CCGT CF: {ccgt_energy / (network.generators.p_nom_opt['ccgt'] * 8760):.1%}")
    print()

    # timeseries of generation
    generation_ts = network.generators_t.p

    # optimal generation capacities
    network.generators.p_nom_opt



    return network


def pypsa_model():

    # Create network with snapshots
    network = pypsa.Network()
    snapshots = pd.date_range("2024-01-01", periods=8760, freq="h")  # Full year
    network.set_snapshots(snapshots)

    # Add bus
    network.add("Bus", "bus_0")

    # Add constant load (100 MW)
    network.add("Load",
                "load",
                bus="bus_0",
                p_set=100)

    # Create realistic wind profile for a year
    np.random.seed(42)
    hours = np.arange(8760)
    wind_profile = np.clip(
        0.35 +  # base capacity factor
        0.15 * np.sin(2 * np.pi * hours / 8760) +  # seasonal variation
        0.20 * np.sin(2 * np.pi * hours / 24) +    # diurnal variation
        0.15 * np.random.randn(8760),              # randomness
        0, 1
    )

    # Add wind farm - capacity is OPTIMIZABLE
    network.add("Generator",
                "wind",
                bus="bus_0",
                p_nom_extendable=True,  # KEY: make capacity a decision variable
                p_nom_min=0,            # minimum capacity
                p_nom_max=500,          # maximum capacity (optional constraint)
                capital_cost=1000000,   # €1M per MW per year (annualized CAPEX)
                marginal_cost=0,        # €0/MWh operational cost
                p_max_pu=wind_profile)

    # Add CCGT - capacity is OPTIMIZABLE
    network.add("Generator",
                "ccgt",
                bus="bus_0",
                p_nom_extendable=True,  # KEY: make capacity a decision variable
                p_nom_min=0,
                p_nom_max=300,
                capital_cost=600000,    # €600k per MW per year (annualized CAPEX)
                marginal_cost=50)       # €50/MWh operational cost

    # Solve for optimal capacities AND dispatch
    print("Solving optimization...")
    t0 = time.time()
    network.optimize()
    t1 = time.time()
    print(f"Optimization completed in {t1 - t0:.2f} seconds")

    # Results
    print("=== OPTIMAL CAPACITIES ===")
    print(f"Wind capacity: {network.generators.p_nom_opt['wind']:.2f} MW")
    print(f"CCGT capacity: {network.generators.p_nom_opt['ccgt']:.2f} MW")

    print("\n=== COSTS ===")
    print(f"Total system cost: €{network.objective:,.2f}/year")

    # Break down costs
    wind_capex = network.generators.p_nom_opt['wind'] * network.generators.capital_cost['wind']
    ccgt_capex = network.generators.p_nom_opt['ccgt'] * network.generators.capital_cost['ccgt']
    wind_opex = (network.generators_t.p['wind'] * network.generators.marginal_cost['wind']).sum()
    ccgt_opex = (network.generators_t.p['ccgt'] * network.generators.marginal_cost['ccgt']).sum()

    print(f"\nWind CAPEX: €{wind_capex:,.2f}/year")
    print(f"Wind OPEX:  €{wind_opex:,.2f}/year")
    print(f"CCGT CAPEX: €{ccgt_capex:,.2f}/year")
    print(f"CCGT OPEX:  €{ccgt_opex:,.2f}/year")

    print("\n=== ENERGY DELIVERED ===")
    wind_energy = network.generators_t.p['wind'].sum()
    ccgt_energy = network.generators_t.p['ccgt'].sum()
    total_demand = 100 * 8760  # MW * hours

    print(f"Wind: {wind_energy:,.0f} MWh ({wind_energy/total_demand:.1%} of demand)")
    print(f"CCGT: {ccgt_energy:,.0f} MWh ({ccgt_energy/total_demand:.1%} of demand)")

    print("\n=== CAPACITY FACTORS ===")
    print(f"Wind CF: {wind_energy / (network.generators.p_nom_opt['wind'] * 8760):.1%}")
    print(f"CCGT CF: {ccgt_energy / (network.generators.p_nom_opt['ccgt'] * 8760):.1%}")
    print()



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
    print(f"kwargs: {kwargs}")
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
    # pypsa_model()
    kwargs = {
        "toggles": {
            "Data Centre Capacity": True,
            "Wind Farm Capacity": True,
            "Solar PV Capacity": True,
            "Grid Connection": True,
            "Gas CCGT Capacity": True,
            "SMR Capacity": True,
            "CO2 price": True,  
        },
        "sliders": {
            "Data Centre Capacity": 50,
            "Wind Farm Capacity": 50,
            "Solar PV Capacity": 50,
            "Grid Connection": 50,
            "Gas CCGT Capacity": 50,
            "SMR Capacity": 50,
            "CO2 price": 50,
        },
        "tiers": {
            "Data Centre Capacity": "Mid",
            "Wind Farm Capacity": "Mid",
            "Solar PV Capacity": "Mid",
            "Grid Connection": "Mid",
            "Gas CCGT Capacity": "Mid",
            "SMR Capacity": "Mid",
            "CO2 price": "Mid"
        }
    }
    results = execute_optimisation(
        toggles=kwargs["toggles"],
        sliders=kwargs["sliders"],
        tiers=kwargs["tiers"],
    )
    print(results)
