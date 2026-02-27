import pypsa
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import time

import calculate_pv_profile
import calculate_wind_profile

logger = logging.getLogger(__name__)

class Generation():
    def __init__(self, name: str, data: dict):
        """
        """
        # TODO - ensure all units are handled in MW

        self.name = name
        self.location = data.get("location", None)

        costs = data["costs"]
        self.capex = costs.get("CAPEX", {}).get("value", 0)
        self.opex_f = costs.get("Fixed OPEX", {}).get("value", 0)
        self.opex_v = costs.get("Variable OPEX", {}).get("value", 0)

        # search for energy cost value
        self.energy_cost = 0
        for keyword in ["Fuel cost", "Electricity Price", "Energy cost"]:
            if keyword in costs:
                self.energy_cost = costs[keyword].get("value", 0)
                break

        if self.name in ["Solar", "Wind"]:
            self.pu_profile = self.get_renewable_profile()
        else:
            self.pu_profile = pd.DataFrame({"pu_power": np.ones(8760)})

        if self.name == "Gas":
            self.carrier = "gas"
            self.efficiency = 0.5  # TODO check assumption
        else:
            self.carrier = None
            self.efficiency = 1

        pass


    def get_renewable_profile(self):

        # read ERA5 parquet file
        # TODO - make loading dynamic based on lat/long
        era5_data = self.get_era5_data()

        if self.name == "Solar":
            # get solar profile
            return calculate_pv_profile.main(
                lat=self.location["lat"],
                lon=self.location["lon"],
                data_source=None,  # TODO - make clearer how to use these arguments
                era5_data=era5_data
                )

        elif self.name == "Wind":
            # get wind profile
            return calculate_wind_profile.main(
                lat=self.location["lat"],
                lon=self.location["lon"],
                onshore=self.location["onshore"],
                era5_df=era5_data
                )

    def get_era5_data(self):
        # read ERA5 parquet file
        _script_dir = Path(__file__).parent
        # go up a level to get to the data directory
        data_dir = _script_dir.parent / "data"
        # read ERA5 parquet file
        era5_data = pd.read_parquet(data_dir / "processed" / "single_point_UK_2025.parquet")
        return era5_data


class Storage():
    def __init__(self, name: str, data: dict):

        self.name = name

        costs = data["costs"]
        self.capex = costs.get("CAPEX", {}).get("value", 0)
        self.opex_f = costs.get("Fixed OPEX", {}).get("value", 0)
        self.opex_v = costs.get("Variable OPEX", {}).get("value", 0)

        # TODO - add these to the input JSON
        self.round_trip_efficiency = data.get("round_trip_efficiency", 0.95)
        self.standing_loss = data.get("standing_loss", 0.001)  # TODO - check assumption
        self.max_hours = data.get("max_hours", 2)


def build_optimiser_results(network):
    """
    Build the optimiser-results-data dict from a solved PyPSA network.
    Structure matches what results_accordion expects (generation_ts, storage_ts,
    generator_stats, storage_stats, total_cost, total_emissions).
    """
    # total_cost: JSON-serialisable float
    total_cost = float(network.objective)

    # generation_ts: dict[str, list[float]]
    generation_ts_df = network.generators_t.p
    generation_ts = {gen: generation_ts_df[gen].tolist() for gen in generation_ts_df.columns}

    # annual generation: dict[str, float]
    annual_generation = generation_ts_df.sum().to_dict()

    # storage_ts: empty in v2 (no storage); same shape as generation_ts if storage exists
    if len(network.storage_units) > 0:
        storage_ts_df = network.storage_units_t.p
        storage_ts = {st: storage_ts_df[st].tolist() for st in storage_ts_df.columns}
    else:
        storage_ts = {}

    # generator_stats: p_nom_opt, capital_cost, marginal_cost (each dict[str, float])
    stats_cols = ["p_nom_opt", "capital_cost", "marginal_cost"]
    raw_gen = network.generators[stats_cols].to_dict()
    # TODO - capex needs to be multiplied by p_nom_opt
    # TODO - capex needs to be separated from opex_f
    # TODO - marginal_cost needs to be separated into opex_v and energy_cost
    # TODO - energy_cost and opex_v need to be multiplied by MWh produced (/ efficiency for energy_cost)
    generator_stats = {
        k: {name: float(v) for name, v in raw_gen[k].items()}
        for k in stats_cols
    }

    # storage_stats: same shape; empty dicts when no storage
    if len(network.storage_units) > 0:
        raw_stor = network.storage_units[stats_cols].to_dict()
        # TODO - capex needs to be multiplied by p_nom_opt
        # TODO - capex needs to be separated from opex_f
        # TODO - marginal_cost needs to be separated into opex_v and energy_cost
        # TODO - energy_cost and opex_v need to be multiplied by MWh produced (/ efficiency for energy_cost)
        storage_stats = {
            k: {name: float(v) for name, v in raw_stor[k].items()}
            for k in stats_cols
        }
    else:
        storage_stats = {k: {} for k in stats_cols}

    # total_emissions: only carriers with co2_emissions (renewables -> 0)
    carrier_emissions = network.generators.carrier.map(
        network.carriers.co2_emissions
    ).fillna(0)
    total_emissions = float(
        network.generators_t.p.multiply(carrier_emissions, axis=1).sum().sum()
    )

    return {
        "total_cost": total_cost,
        "total_emissions": total_emissions,
        "generator_stats": generator_stats,
        "storage_stats": storage_stats,   
        "annual_generation": annual_generation,
        "generation_ts": generation_ts,
        "storage_ts": storage_ts,
    }


def main(data: dict):
    # all units in MW
    load = data["data_centre_capacity"]

    gen = data["generation"]
    gen_enabled = [g for g in gen if gen[g]["enabled"]]
    generation_instances = [Generation(g, gen[g]) for g in gen_enabled]

    storage = data["battery_storage"]
    storage_enabled = storage["enabled"]
    if storage_enabled:
        storage_instances = [Storage("battery_storage", storage)]
    else:
        storage_instances = []

    network = pypsa_model(load=load, generators=generation_instances, storage=storage_instances)
    results = build_optimiser_results(network)
    return results

def pypsa_model(load: float, generators: list[Generation], storage: list[Storage]):

    # Create network with snapshots
    network = pypsa.Network()
    snapshots = pd.date_range("2024-01-01", periods=8760, freq="h")  # Full year
    network.set_snapshots(snapshots)

    # Add bus
    network.add("Bus", "bus_0")

    # TODO - check assumption
    network.add("Carrier", "gas", co2_emissions=0.35)  # tonnes/MWh for CCGT

    # Add constant load (100 MW)
    network.add(
        "Load",
        name="Data centre load",
        bus="bus_0",
        p_set=load
    )

    for gen in generators:
        # Add generator - capacity is OPTIMIZABLE
        print("Adding generator: ", gen.name)
        print("Capex: ", gen.capex)
        print("Opex_v: ", gen.opex_v)
        print("Energy cost: ", gen.energy_cost)
        print()
        marginal_cost = gen.opex_v + gen.energy_cost
        capex = gen.capex + gen.opex_f
        # TODO - annualise the capex
        network.add(
            "Generator",
            name=gen.name,
            bus="bus_0",
            carrier=gen.carrier,
            p_nom_extendable=True,  # KEY: make capacity a decision variable
            p_nom_min=0,            # minimum capacity
            capital_cost=capex,     # cost per MW per year (annualized CAPEX)
            marginal_cost=marginal_cost,
            efficiency=gen.efficiency,
            p_max_pu=gen.pu_profile["pu_power"].values
            )

    for store in storage:
        one_way_efficiency = np.sqrt(store.round_trip_efficiency)
        capex = store.capex + store.opex_f
        # TODO - annualise the capex
        # OR use discount rate, overnight cost and lifetime
        network.add(
            "StorageUnit",
            name=store.name,
            bus="bus_0",
            p_nom_extendable=True,  # KEY: make capacity a decision variable
            p_nom_min=0,            # minimum capacity
            capital_cost=capex,     # cost per MW per year (annualized CAPEX)
            marginal_cost=store.opex_v,
            max_hours=store.max_hours,
            standing_loss=store.standing_loss,
            efficiency_store=one_way_efficiency,
            efficiency_dispatch=one_way_efficiency,
        )

    # Solve for optimal capacities AND dispatch
    logger.info("Solving optimization...")
    t0 = time.time()
    network.optimize()
    t1 = time.time()
    logger.info(f"Optimization completed in {t1 - t0:.2f} seconds")

    # Results
    print("=== OPTIMAL CAPACITIES ===")
    for gen in generators:
        print(f"{gen.name} capacity: {network.generators.p_nom_opt[gen.name]:.2f} MW")

    for store in storage:
        print(f"{store.name} capacity: {network.storage_units.p_nom_opt[store.name]:.2f} MW")

    print("\n=== COSTS ===")
    print(f"Total system cost: €{network.objective:,.2f}/year")

    return network


if __name__ == "__main__":
    _script_dir = Path(__file__).parent
    test_json = _script_dir / "collected_parameters.json"
    # read test json
    with open(test_json, "r") as f:
        test_data = json.load(f)
    print(test_data)
    main(test_data)