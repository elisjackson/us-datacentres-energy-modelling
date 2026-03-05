import functools
import pypsa
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import json
from pathlib import Path
import logging
import time
import boto3

try:
    import src_plotly.calculate_pv_profile as calculate_pv_profile
    import src_plotly.calculate_wind_profile as calculate_wind_profile
except ImportError:
    import calculate_pv_profile
    import calculate_wind_profile

logger = logging.getLogger(__name__)

MODE = "aws" # "local" or "aws"
S3_BUCKET = "datacentres-dev-data-207662791637"
S3 = boto3.client("s3")


@functools.cache
def _get_s3_lat_lon_index() -> list[tuple[float, float]]:
    """List all available (lat, lon) coordinate pairs from the S3 ERA5 partition index.

    Cached so the S3 listing is only performed once per script execution.
    """
    logger.info("Getting S3 lat/lon indexes")
    print(f"Getting S3 lat/lon indexes from {S3_BUCKET}")
    response = S3.list_objects_v2(Bucket=S3_BUCKET, Prefix="era5/")
    objects = response.get("Contents", [])
    coords = []
    for obj in objects:
        key = obj["Key"]
        if ".parquet" not in key:
            continue
        lat_val = float(key.split("/")[-3].split("=")[1])
        lon_val = float(key.split("/")[-2].split("=")[1])
        coords.append((lat_val, lon_val))
    return coords


@functools.cache
def _get_era5_data_s3(lat: float, lon: float) -> pd.DataFrame:
    """Fetch ERA5 data for the nearest available grid point to (lat, lon) from S3.

    Cached per unique (lat, lon) pair so repeated calls for the same location
    do not trigger additional S3 reads.
    """
    coords = _get_s3_lat_lon_index()
    lat_vals = np.array([c[0] for c in coords])
    lon_vals = np.array([c[1] for c in coords])
    distances = np.sqrt((lat_vals - lat) ** 2 + (lon_vals - lon) ** 2)
    closest_lat, closest_lon = coords[np.argmin(distances)]
    logger.info(f"Reading ERA5 data from S3 for nearest point ({closest_lat}, {closest_lon})")
    dataset = pa_ds.dataset(
        f"s3://{S3_BUCKET}/era5/",
        format="parquet",
        partitioning=pa_ds.partitioning(
            pa.schema([
                ("country", pa.string()),
                ("lat", pa.float32()),
                ("lon", pa.float32()),
            ]),
            flavor="hive",
        ),
    )
    return dataset.to_table(
        filter=(
            (pc.field("country") == "United Kingdom")
            & (pc.field("lat") == np.float32(closest_lat))
            & (pc.field("lon") == np.float32(closest_lon))
        )
    ).to_pandas()


class Generation():
    def __init__(self, name: str, data: dict):
        """
        """

        self.name = name
        self.location = data.get("location", None)

        costs = data["costs"]
        
        capex_dict = costs.get("CAPEX", {})
        self.capex_unit = capex_dict.get("unit", "")
        if "/kW" not in self.capex_unit or "/kWh" in self.capex_unit:
            raise ValueError(f"Expecting CAPEX in /kW value, got {self.capex_unit}")
        self.capex = capex_dict.get("value", 0) * 1000  # convert to /MW cost

        opex_f_dict = costs.get("Fixed OPEX", {})
        self.opex_f_unit = opex_f_dict.get("unit", "")
        if opex_f_dict:
            if "/kW" not in self.opex_f_unit or "/kWh" in self.opex_f_unit:
                raise ValueError(f"Expecting Fixed OPEX in /kW value, got {self.opex_f_unit}")
        self.opex_f = opex_f_dict.get("value", 0) * 1000  # convert to /MW cost
        
        opex_v_dict = costs.get("Variable OPEX", {})
        self.opex_v_unit = opex_v_dict.get("unit", "")
        if opex_v_dict:
            if "/MWh" not in self.opex_v_unit:
                raise ValueError(f"Expecting Variable OPEX in /MWh value, got {self.opex_v_unit}")
        self.opex_v = opex_v_dict.get("value", 0)

        # search for energy cost value
        self.energy_cost = 0
        for keyword in ["Fuel cost", "Electricity Price", "Energy cost", "Power cost"]:
            if keyword in costs:
                self.energy_cost_unit = costs[keyword].get("unit", "")
                if "/MWh" not in self.energy_cost_unit:
                    raise ValueError(f"Expecting Energy cost in /MWh value, got {self.energy_cost_unit}")
                self.energy_cost = costs[keyword].get("value", 0)
                break

        if self.name in ["Solar", "Wind"]:
            self.pu_profile = self.get_renewable_profile(
                lat=self.location["lat"],
                lon=self.location["lon"]
                )
        else:
            self.pu_profile = pd.DataFrame({"pu_power": np.ones(8760)})

        if self.name == "Gas":
            self.carrier = "gas"
        else:
            self.carrier = None

        self.efficiency = costs.get("Efficiency", {}).get("value", 100) / 100
        self.lifetime = costs.get("Lifetime", {}).get("value", 20)


    def get_renewable_profile(self, lat: float, lon: float):

        # read ERA5 parquet file
        era5_data = self.get_era5_data(lat, lon)

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

    def get_era5_data(self, lat: float, lon: float):
        # read ERA5 parquet file
        if MODE == "local":
            _script_dir = Path(__file__).parent
            # go up a level to get to the data directory
            data_dir = _script_dir.parent / "data"
            # read ERA5 parquet file
            era5_data = pd.read_parquet(data_dir / "processed" / "single_point_UK_2025.parquet")
        else:
            era5_data = _get_era5_data_s3(lat, lon)
        return era5_data


class Storage():
    def __init__(self, name: str, data: dict):

        self.name = name

        costs = data["costs"]

        capex_dict = costs.get("CAPEX", {})
        self.capex_unit = capex_dict.get("unit", "")
        if "/kW" not in self.capex_unit or "/kWh" in self.capex_unit:
            raise ValueError(f"Expecting CAPEX in /kW value, got {self.capex_unit}")
        self.capex = capex_dict.get("value", 0) * 1000  # convert to /MW cost
        
        opex_f_dict = costs.get("Fixed OPEX", {})
        self.opex_f_unit = opex_f_dict.get("unit", "")
        if opex_f_dict:
            if "/kW" not in self.opex_f_unit or "/kWh" in self.opex_f_unit:
                raise ValueError(f"Expecting Fixed OPEX in /kW value, got {self.opex_f_unit}")
        self.opex_f = opex_f_dict.get("value", 0) * 1000  # convert to /MW cost

        opex_v_dict = costs.get("Variable OPEX", {})
        self.opex_v_unit = opex_v_dict.get("unit", "")
        if opex_v_dict:
            if "/MWh" not in self.opex_v_unit:
                raise ValueError(f"Expecting Variable OPEX in /MWh value, got {self.opex_v_unit}")

        self.round_trip_efficiency = costs.get("Round trip efficiency", {}).get("value", 85) / 100
        self.standing_loss = costs.get("standing_loss", 0)
        self.max_hours = costs.get("Energy/Power ratio", {}).get("value", 4)
        self.lifetime = costs.get("Lifetime", {}).get("value", 20)


def build_optimiser_results(load, network, generation_instances, storage_instances, co2_price):
    """
    Build the optimiser-results-data dict from a solved PyPSA network.
    Structure matches what results_accordion expects (generation_ts, storage_ts,
    generator_stats, storage_stats, total_cost, total_emissions).
    """
    # total_cost: JSON-serialisable float
    total_cost = float(network.objective)

    # build dataframe of the PyPSA generator inputs
    gen_inputs_df = pd.DataFrame()
    for gen in generation_instances:
        attributes = gen.__dict__
        if "pu_profile" in attributes:
            attributes.pop("pu_profile")  # keep only scalar attributes
        gen_inputs_df = pd.concat([gen_inputs_df, pd.DataFrame(attributes, index=[gen.name])])
    # drop anything we don't need
    # e.g. carrier and efficiency are already given in the PyPSA network.generators dataframe
    gen_inputs_df.drop(columns=["name", "location", "carrier", "efficiency", "lifetime"], inplace=True)
    gen_inputs_df.rename(columns={"capex": "capex_pu", "opex_f": "opex_f_pu"}, inplace=True)

    # build dataframe of the PyPSA storage inputs
    storage_inputs_df = pd.DataFrame()
    for store in storage_instances:
        attributes = store.__dict__
        storage_inputs_df = pd.concat([storage_inputs_df, pd.DataFrame(attributes, index=[store.name])])
    # drop anything we don't need
    # e.g. carrier and efficiency are already given in the PyPSA network.storage_units dataframe
    storage_inputs_df.drop(
        columns=["name", "round_trip_efficiency", "standing_loss", "max_hours", "lifetime"],
        inplace=True,
        errors="ignore"
        )
    storage_inputs_df.rename(
        columns={"capex": "capex_pu", "opex_f": "opex_f_pu"},
        inplace=True,
        errors="ignore"
        )

    # generation_ts: dict[str, list[float]]
    generation_ts_df = network.generators_t.p
    generation_ts = {gen: generation_ts_df[gen].tolist() for gen in generation_ts_df.columns}

    # annual generation: dict[str, float]
    annual_generation_df = pd.DataFrame(generation_ts_df.sum().rename("annual_generation"))
    annual_generation = annual_generation_df.to_dict()

    # storage_ts: empty in v2 (no storage); same shape as generation_ts if storage exists
    if len(network.storage_units) > 0:
        storage_ts_df = network.storage_units_t.p
        storage_ts = {st: storage_ts_df[st].tolist() for st in storage_ts_df.columns}
        # annual storage flow - use as sanity check
        annual_storage_flow = storage_ts_df.sum().to_dict()
    else:
        storage_ts = {}

    # generator_stats: p_nom_opt, capital_cost, marginal_cost (each dict[str, float])
    gen_stats_df = network.generators
    gen_stats_df = pd.merge(
        gen_stats_df, annual_generation_df, left_index=True, right_index=True
        )
    # merge the PyPSA inputs
    gen_stats_df = pd.merge(
        gen_stats_df, gen_inputs_df, left_index=True, right_index=True
        )
    # calcaulate final results
    # capex is annualised by lifetime
    gen_stats_df["total_capex"] = (
        gen_stats_df["capex_pu"] * gen_stats_df["p_nom_opt"] / gen_stats_df["lifetime"]
        )
    gen_stats_df["total_opex_f"] = gen_stats_df["opex_f_pu"] * gen_stats_df["p_nom_opt"]
    gen_stats_df["total_energy_cost"] = (
        gen_stats_df["energy_cost"] * gen_stats_df["annual_generation"] / gen_stats_df["efficiency"]
        )
    gen_stats_df["total_opex_v"] = gen_stats_df["opex_v"] * gen_stats_df["annual_generation"]
    gen_stats_df["total_opex"] = gen_stats_df["total_opex_f"] + gen_stats_df["total_opex_v"]
    # Calculate CO2 emissions and cost
    gen_stats_df = pd.merge(
        gen_stats_df,
        network.carriers["co2_emissions"],
        left_on="carrier",
        right_index=True,
        how="outer"
        ).rename(columns={"co2_emissions": "co2_emissions_primary"}).fillna(0)
    # total CO₂ emission: tCO2
    gen_stats_df["total_co2_emission"] = (
        gen_stats_df["co2_emissions_primary"]
        * gen_stats_df["annual_generation"]
        / gen_stats_df["efficiency"]
    )
    # co2_price: USD/tCO2
    gen_stats_df["total_co2_cost"] = (
        gen_stats_df["total_co2_emission"] * co2_price
    )
    
    stats_cols = [
        "p_nom_opt",
        "total_capex",
        "total_opex",
        "total_energy_cost",
        "total_co2_emission",
        "total_co2_cost",
    ]
    gen_stats_df = gen_stats_df[stats_cols]
    gen_stats_dict = gen_stats_df.to_dict()

    # storage_stats: same shape; empty dicts when no storage
    if len(network.storage_units) > 0:
        storage_stats_df = network.storage_units
        # merge the PyPSA inputs
        storage_stats_df = pd.merge(
            storage_stats_df, storage_inputs_df, left_index=True, right_index=True
            )
        # capex is annualised by lifetime
        storage_stats_df["total_capex"] = (
            storage_stats_df["capex_pu"] * storage_stats_df["p_nom_opt"] / storage_stats_df["lifetime"]
            )
        storage_stats_df["total_opex_f"] = storage_stats_df["opex_f_pu"] * storage_stats_df["p_nom_opt"]
        storage_stats_df["total_opex"] = storage_stats_df["total_opex_f"]

        stats_cols = [
            "p_nom_opt",
            "total_capex",
            "total_opex",
        ]
        storage_stats_df = storage_stats_df[stats_cols]
        storage_stats_dict = storage_stats_df.to_dict()
    else:
        storage_stats_dict = {k: {} for k in stats_cols}

    return {
        "status": "ok",
        "load": load,
        "total_cost": total_cost,
        "total_emissions": gen_stats_df["total_co2_emission"].sum(),
        "generator_stats": gen_stats_dict,
        "storage_stats": storage_stats_dict,
        "total_generation_capacity": gen_stats_df["p_nom_opt"].sum(),
        "annual_generation": annual_generation["annual_generation"],
        "generation_ts": generation_ts,
        "storage_ts": storage_ts,
    }


def main(data: dict):
    # all units in MW
    load = data["data_centre_capacity"]

    gen = data["generation"]
    gen_enabled = [g for g in gen if gen[g]["enabled"]]
    generation_instances = [Generation(g, gen[g]) for g in gen_enabled]

    storage = data.get("battery_storage", {})
    storage_enabled = storage["enabled"]
    if storage_enabled:
        storage_instances = [Storage("Battery storage", storage)]
    else:
        storage_instances = []

    co2_price = (
        (
            (data.get("co2") or {}).get("costs") or {}).get("Carbon price") or {}
        ).get("value", 0.0)

    network = pypsa_model(
        load=load,
        generators=generation_instances,
        storage=storage_instances,
        co2_price=co2_price
        )
    if network.model.termination_condition == "infeasible":
        return {"status": "infeasible"}
    results = build_optimiser_results(
        load,
        network,
        generation_instances,
        storage_instances,
        co2_price
        )
    return results


def pypsa_model(
    load: float,
    generators: list[Generation],
    storage: list[Storage],
    co2_price: float
    ):
    """
    co2_price: float - price of CO2 in USD/tCO2
    """

    # Create network with snapshots
    network = pypsa.Network()
    snapshots = pd.date_range("2024-01-01", periods=8760, freq="h")  # Full year
    network.set_snapshots(snapshots)

    # Add bus
    network.add("Bus", "bus_0")

    gas_co2_emissions = 0.2  # tCO2/MWh primary energy
    network.add("Carrier", "gas", co2_emissions=gas_co2_emissions)
    
    # Add constant load
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
        marginal_cost = gen.opex_v + (gen.energy_cost / gen.efficiency)
        if gen.carrier == "gas":
            # co2_price: USD/tCO2
            # gas_co2_emissions: tCO2/MWh primary energy
            # efficiency: %
            # marginal_cost: USD/MWh
            marginal_cost += gas_co2_emissions * co2_price / gen.efficiency
        capex = (gen.capex / gen.lifetime) + gen.opex_f

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
            p_max_pu=gen.pu_profile["pu_power"].values,
            lifetime=gen.lifetime,
            )

    for store in storage:
        one_way_efficiency = np.sqrt(store.round_trip_efficiency)
        capex = (store.capex / store.lifetime) + store.opex_f
        network.add(
            "StorageUnit",
            name=store.name,
            bus="bus_0",
            p_nom_extendable=True,  # KEY: make capacity a decision variable
            p_nom_min=0,            # minimum capacity
            capital_cost=capex,     # cost per MW per year (annualized CAPEX)
            # marginal_cost=store.opex_v,
            max_hours=store.max_hours,
            standing_loss=store.standing_loss,
            efficiency_store=one_way_efficiency,
            efficiency_dispatch=one_way_efficiency,
            lifetime=store.lifetime,
        )

    # Solve for optimal capacities AND dispatch
    logger.info("Solving optimization...")
    t0 = time.time()
    network.optimize()
    t1 = time.time()
    logger.info(f"Optimization completed in {t1 - t0:.2f} seconds")

    return network


if __name__ == "__main__":
    _script_dir = Path(__file__).parent
    test_json = _script_dir / "collected_parameters_infeas.json"
    # read test json
    with open(test_json, "r") as f:
        test_data = json.load(f)
    print(test_data)
    main(test_data)