import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Literal


def extrapolate_wind_profile(
    target_height: float,
    v1: pd.Series,
    h1: float,
    wt_type: Literal["onshore", "offshore"]
) -> pd.Series:
    """
    On the ground, the wind is strongly braked by obstacles and surface roughness.
    High above the ground in the undisturbed air layers of the
    geostrophic wind (at approx. 5 km above ground) the wind is no longer
    influenced by the surface.
    
    Between these two extremes, wind speed changes with height.
    This phenomenon is called vertical wind shear.
    
    In flat terrain and with a neutrally stratisfied atmosphere,
    the logarithmic wind profile is a good estimation for the vertical wind shear:

    v2 = v1 * (np.log(h2 / z0) / np.log(h1 / z0))
    
    The reference wind speed v1 is measured at height h1.
    v2 is the wind speed at height h2. z0 is the roughness length (see table above).
    """

    if wt_type == "onshore":
        # TODO - check assumption
        z0 = 0.1
    elif wt_type == "offshore":
        # TODO - check assumption
        z0 = 0.03

    ratio = np.log(target_height / z0) / np.log(h1 / z0)
    wind_speed = v1.mul(ratio)
    return wind_speed


def main():

    # Read ERA5 data
    data_path = Path("data/processed/single_point_UK_2025.parquet")

    if not data_path.exists():
        raise FileNotFoundError(
            f"ERA5 data file not found: {data_path}\n"
            "Run meteo_data/split_into_parts.py first to generate the single-point data."
        )

    # Read the parquet file
    era5_df = pd.read_parquet(data_path)

    # extrapolate 100m wind speed to the hub height
    hub_height = 140
    era5_df["wind_speed_hh"] = extrapolate_wind_profile(
        hub_height,
        era5_df["wind_speed_100"],
        100,
        "onshore"
        )
    era5_df = era5_df[["wind_speed_hh"]]
    # round to nearest 0.5
    era5_df["wind_speed_hh"] = (2 * era5_df["wind_speed_hh"]).round() / 2


    wt_type = "onshore"
    wt = "Vestas V112-3.3"
    # get parent path
    parent_path = Path(__file__).parent
    power_curves_path = parent_path / "power_curves" / "power_curves.json"
    with open(power_curves_path, "r") as f:
        power_curves = json.load(f)
    power_curve_data = power_curves[wt_type][wt]
    wind_speed = power_curve_data["wind_speed"]
    power = power_curve_data["power"]
    power_df = pd.DataFrame({"wind_speed": wind_speed, "power": power})

    # map wind speed to power
    era5_df = era5_df.merge(
        power_df,
        left_on="wind_speed_hh",
        right_on="wind_speed",
        how="left"
        ).fillna(0)

    annual_generation = era5_df["power"].sum()  # kWh

    # add losses
    wake_losses = 0.065
    electrical_losses = 0.01
    availability_losses = 0.03
    # https://energiaa.vamk.fi/en/articles/komptence/uncertainties-in-wind-energy-production/
    total_losses = (1 - wake_losses) * (1 - electrical_losses) * (1 - availability_losses)
    annual_generation = annual_generation * (total_losses)
    capacity_factor = annual_generation / (power_curve_data["capacity"] * 8760)

    print(f"Annual generation: {annual_generation:,.0f} kWh")
    print(f"Capacity factor: {capacity_factor:.1%}")

    era5_df["pu_power"] = era5_df["power"] / power_curve_data["capacity"]

    return era5_df[["pu_power"]]

if __name__ == "__main__":
    pu_power = main()