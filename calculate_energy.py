import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time
import pv_model
from pathlib import Path

def custom_round(x, base=5):
    return base * round(float(x)/base)

def solve(
    target_capacity: float,
    wind_mw: float,
    pv_mw: float,
    bess_mw: float,
    bess_mwh: float,
    mean_ws: float = 12,
    mean_ghi: float = 1900,
    location: tuple[int] = (52.5, 13.4, 34),
    rename_cols_for_data_centre_application: bool = False
):
    """
    location: latitude, longitude, altitude. co-ordinates in WGS84
    """
    print(f"Calculating energy for loc: {location}...")
    t0 = time.time()
    # get weather file
    df = get_weather_file()

    
    # datetime column; arbitrary year value
    df["datetime"] = pd.to_datetime({
        "year": 2024,
        "month": df["Month"],
        "day": df["Day"],
        "hour": df["Hour"]
    })
    df = df.drop(columns=["Month", "Day", "Hour"])
    # re-order columns
    df = df[["datetime"] + [c for c in df.columns if c != "datetime"]]

    # initialise demand
    df["demand"] = target_capacity

    # scale wind speed and GHI
    df = scale_wind_speed(df, mean_ws, mean_ghi)

    # get power curve
    power_curve = wt_power_curve(wind_mw)

    # calculate wind profile
    # round to nearest 0.25 for merge preparation
    df["Wind Speed"] = df["Wind Speed"].apply(
        lambda x: custom_round(x, base=0.25)
        )
    # merge power curve
    df = pd.merge(
        df,
        power_curve,
        left_on="Wind Speed",
        right_on="wind_speed",
        how="left"
        ).fillna(0)

    # calculate PV profile
    pv_ac = pv_model.calculate(location, pv_mw)
    pv_ac = pv_ac.reset_index(drop=True)
    pv_ac.name = "PV_AC"
    df = pd.merge(
        df,
        pv_ac,
        how="left",
        left_index=True,
        right_index=True
    ).fillna(0)
    df["PV_AC"] = df["PV_AC"].clip(lower=0)

    # generation sum
    df["generation"] = df["farm_wind_power"] + df["PV_AC"]
    df["excess_gen"] = (df["generation"] - target_capacity).clip(lower=0)
    df["demand_after_generation"] = (df["demand"] - df["generation"]).clip(lower=0)

    # add battery storage profile
    df = calculate_battery_profile(df, bess_mw, bess_mwh)
    df["gen_plus_pv"] = df["generation"] + df["bess_gen"]

    # calculate top-up gas
    df["demand_after_gen_plus_pv"] = (
        df["demand"] - df["gen_plus_pv"]
    ).clip(lower=0)
    df["gas_demand"] = df["demand_after_gen_plus_pv"] / 0.9

    # get GNESTE assumptions
    df_GNESTE = pd.concat([
        get_GNESTE_assumptions(energy_tech="Gas CCGT"),
        get_GNESTE_assumptions(energy_tech="BESS"),
        get_GNESTE_assumptions(energy_tech="Wind"),
        get_GNESTE_assumptions(energy_tech="Solar PV")
    ]).set_index("energy_tech")

    # calculate CAPEX costs
    gas_mw = df["gas_demand"].max()
    capex = {
        "Gas CCGT": df_GNESTE.loc["Gas CCGT", "CAPEX"] * gas_mw,
        "BESS": df_GNESTE.loc["BESS", "CAPEX"] * bess_mw,
        "Wind": df_GNESTE.loc["Wind", "CAPEX"] * wind_mw,
        "Solar PV": df_GNESTE.loc["Solar PV", "CAPEX"] * pv_mw,
    }

    # calculate OPEX costs
    opex = {
        "Gas CCGT": df_GNESTE.loc["Gas CCGT", "OPEX"] * gas_mw,
        "BESS": df_GNESTE.loc["BESS", "OPEX"] * bess_mw,
        "Wind": df_GNESTE.loc["Wind", "OPEX"] * wind_mw,
        "Solar PV": df_GNESTE.loc["Solar PV", "OPEX"] * pv_mw,
    }

    # calculate gas energy costs
    gas_fuel_cost = {
        "Gas CCGT": df_GNESTE.loc["Gas CCGT", "fuel_price"] * df["gas_demand"].sum()
    }

    # Combine into one DataFrame
    df_costs_emissions = pd.DataFrame({
        "CAPEX": capex,
        "OPEX": opex,
        "Fuel cost": gas_fuel_cost
    }).fillna(0)

    # sum lifetime costs, assume 20 year lifetime
    lifetime_assumed = 20
    df_costs_emissions["20yr OPEX"] = df_costs_emissions["OPEX"] * lifetime_assumed
    df_costs_emissions["20yr fuel cost"] = df_costs_emissions["Fuel cost"] * lifetime_assumed

    df_costs_emissions["Lifetime cost"] = (
        df_costs_emissions["CAPEX"]
        + df_costs_emissions["20yr OPEX"]
        + df_costs_emissions["20yr fuel cost"]
        )

    df_costs_emissions.loc["Gas CCGT", "kgCO2"] = 202 * df["gas_demand"].sum()
    df_costs_emissions["Lifetime kgCO2"] = lifetime_assumed * df_costs_emissions["kgCO2"]
    df_costs_emissions = df_costs_emissions.fillna(0)

    if rename_cols_for_data_centre_application:
        df = df.rename(columns={
            "demand": "Data centre demand",
            "farm_wind_power": "Wind generation",
            "PV_AC": "Solar PV generation",
            "gas_demand": "Gas consumption",
            "bess_power": "BESS power",
        })

    # return results
    print(f"Calculated energy in {round(time.time() - t0, 3)}s")
    return df, df_costs_emissions

def calculate_battery_profile(
    df: pd.DataFrame,
    bess_mw: float,
    bess_mwh: float
    ) -> pd.DataFrame:
    t0 = time.time()    

    gen = np.array(df["generation"])
    demand = np.array(df["demand"])

    # 1-way efficiency = sqrt(round trip efficiency)
    efficiency = np.sqrt(0.95)

    soc_min = 0

    # power "into" battery will be +ve, "out of" will be -ve
    bess_power = np.zeros_like(gen, dtype=float)
    soc = np.zeros_like(gen, dtype=float)

    for t in range(1, len(soc)):

        # charge if generation is more than demand
        if gen[t] > demand[t]:
            bess_power[t] = efficiency * (gen[t] - demand[t])

        # discharge if demand is more than generation
        elif gen[t] < demand[t]:
            bess_power[t] = -1 * (demand[t] - gen[t]) / efficiency

        # clip power to MW capacity
        bess_power[t] = np.clip(bess_power[t], -bess_mw, bess_mw)

        # clip power to SOC limits
        soc_charge_headroom = bess_mwh - soc[t-1]
        soc_discharge_headroom = soc[t-1] - soc_min
        bess_power[t] = np.clip(
            bess_power[t],
            -soc_discharge_headroom,
            soc_charge_headroom
            )

        # update SOC
        soc[t] = soc[t-1] + bess_power[t]

    df["bess_power"] = bess_power
    df["bess_soc"] = soc
    # power 'into' data centre
    df["bess_gen"] = -1 * df["bess_power"].clip(upper=0) * efficiency

    print(f"BESS calculations completed in {round(time.time() - t0, 3)}s")
    
    return df


def get_weather_file() -> pd.DataFrame:
    """
    Load typical year weatherfile
    """
    fpath = Path("inputs") / "weather_file" / "726234_41.65_-95.74_tmy-2024.csv"
    df = pd.read_csv(fpath, header=2)
    df = df.drop(columns=["Year", "Minute"])
    return df

def scale_wind_speed(df: pd.DataFrame, mean_ws: float, mean_ghi: float):
    
    mean_wf_ws = df["Wind Speed"].mean()
    wf_scale_ratio = mean_ws / mean_wf_ws
    df["Wind Speed"] = df["Wind Speed"] * wf_scale_ratio
    return df

def wt_power_curve(wind_farm_mw):
    """
    wind_farm_mw: the total power of the wind farm in MW

    Power curve data source: https://en.wind-turbine-models.com/powercurves
    Based on Vestas V164-8.0
    """

    power = {
        10: 5.6,
        11: 7.1,
        12: 7.8,
        13: 8.0,
        20: 8.0
    }

    power_curve_df = pd.DataFrame({
        "wind_speed": list(power.keys()),
        "power": list(power.values())
    })
    power_curve_df = power_curve_df.sort_values("wind_speed").reset_index(drop=True)

    # fine-grained wind speeds from min to max
    fine_wind_speeds = np.linspace(
        power_curve_df.wind_speed.min(),
        power_curve_df.wind_speed.max(),
        num=(len(power_curve_df.wind_speed)-1)*10 + 1
    )

    # Interpolated power values
    interp_func = interp1d(
        power_curve_df.wind_speed,
        power_curve_df.power,
        kind="linear"
    )
    fine_powers = interp_func(fine_wind_speeds)

    # wind_power_single = single wind turbine power output (MW)
    power_curve_df = pd.DataFrame({
        "wind_speed": fine_wind_speeds,
        "wind_power_single": fine_powers
    })

    power_curve_df["farm_wind_power"] = (
        power_curve_df["wind_power_single"] * wind_farm_mw / power_curve_df["wind_power_single"].max()
    )

    return power_curve_df

def get_GNESTE_assumptions(energy_tech: str) -> pd.DataFrame:
    """
    Read mean GNESTE data values from source

    Returns pd.DataFrame with columns
        "CAPEX" (USD/MW)
        "OPEX" (USD/MW)
        "fuel_price" (USD/MWh)
    """
    if energy_tech == "Gas CCGT":
        file = Path("GNESTE_assumptions") / "GNESTE_Gas_Power.csv"
    elif energy_tech == "BESS":
        file = Path("GNESTE_assumptions") / "GNESTE_Battery_Storage.csv"
    elif energy_tech == "Wind":
        file = Path("GNESTE_assumptions") / "GNESTE_Wind_Power.csv"
    elif energy_tech == "Solar PV":
        file = Path("GNESTE_assumptions") / "GNESTE_Solar_Power.csv"
    else:
        raise ValueError("Type not supported")

    df = pd.read_csv(file)

    # year to average data across
    years = ["2020", "2021", "2022", "2023", "2024", "2025"]
    df = df[df["Country"] == "United States of America"]

    # capex (USD/kW)
    capex = df[
        (df["Variable"] == "Capital Cost")
        & (df["Unit"] == "USD/kW")
        ]
    capex = capex[years]
    # mean across rows (different sources / assumptions) and years
    capex = capex.mean().mean()
    # convert to USD/MW
    capex = capex * 1000

    # opex (USD/kW)
    opex = df[
        ((df["Variable"] == "Fixed O&M") if energy_tech != "Solar PV" else (df["Variable"] == "Total O&M"))
        & (df["Unit"] == "USD/kW/yr")
        ]
    opex = opex[years]
    # mean across rows (different sources / assumptions) and years
    opex = opex.mean().mean()
    opex = opex * 1000

    # fuel price (USD/MWh)
    fuel_price = df[df["Variable"] == "Fuel Price"]
    fuel_price = fuel_price[years]
    # mean across rows (different sources / assumptions) and years
    fuel_price = fuel_price.mean().mean()

    results_df = pd.DataFrame({
        "energy_tech": [energy_tech],
        "CAPEX": [capex],
        "OPEX": [opex],
        "fuel_price": [fuel_price]
    })

    return results_df

if __name__ == "__main__":
    df_timeseries, df_costs_emissions = solve(
        target_capacity=10,
        wind_mw=10,
        pv_mw=10,
        bess_mw=2,
        bess_mwh=5,
        rename_cols_for_data_centre_application=True
    )