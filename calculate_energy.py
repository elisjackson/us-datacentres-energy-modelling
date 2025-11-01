import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time

def custom_round(x, base=5):
    return base * round(float(x)/base)

def calculate_energy_func(
    target_capacity: float,
    wind_mw: float,
    pv_mw: float,
    bess_mw: float,
    bess_mwh: float,
    heat_network_potential: float,
    mean_ws: float = 12,
    mean_ghi: float = 1900,
):
    print("Calculating energy...")
    t0 = time.time()
    # get weather file
    df = get_weather_file()

    # scale wind speed and GHI
    df = scale_weather(df, mean_ws, mean_ghi)

    # get power curve
    power_curve = wt_power_curve()

    # calculate wind profile
    df["Wind Speed"] = df["Wind Speed"].apply(
        lambda x: custom_round(x, base=0.25)
        )
    df = pd.merge(
        df,
        power_curve,
        left_on="Wind Speed",
        right_on="wind_speed",
        how="left"
        ).fillna(0)

    # calculate PV profile
    pass

    # add battery storage profile

    # calculate top-up gas

    # get CAPEX costs

    # get OPEX costs

    # get energy costs

    # sum lifetime costs

    # return results
    print(f"Calculated energy in {round(time.time() - t0, 3)}s")
    return {
        "target_capacity": target_capacity,
        "wind_mw": wind_mw,
        "pv_mw": pv_mw,
    }

def calculate_wind_power():
    pass

def calculate_pv_power():
    pass

def get_weather_file() -> pd.DataFrame:
    """
    Load typical year weatherfile
    """
    fpath = r"inputs\weather_file\726234_41.65_-95.74_tmy-2024.csv"
    df = pd.read_csv(fpath, header=2)
    df = df.drop(columns=["Year", "Minute"])
    return df

def scale_weather(df: pd.DataFrame, mean_ws: float, mean_ghi: float):
    
    mean_wf_ws = df["Wind Speed"].mean()
    mean_wf_ghi = df["GHI"].mean()

    wf_scale_ratio = mean_ws / mean_wf_ws
    ghi_scale_ratio = mean_ghi / mean_wf_ghi

    df["Wind Speed"] = df["Wind Speed"] * wf_scale_ratio
    df["GHI"] = df["GHI"] * ghi_scale_ratio
    
    return df

def wt_power_curve():
    """
    Data source: https://en.wind-turbine-models.com/powercurves
    Vestas V164-8.0
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

    power_curve_df = pd.DataFrame({
        "wind_speed": fine_wind_speeds,
        "wind_power": fine_powers
    })

    return power_curve_df

if __name__ == "__main__":
    calculate_energy_func(
        target_capacity=10,
        wind_mw=10,
        pv_mw=10,
        bess_mw=10,
        bess_mwh=10,
        heat_network_potential=10,
    )