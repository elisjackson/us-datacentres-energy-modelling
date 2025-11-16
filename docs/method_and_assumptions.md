### Dashboard map data
- **Wind speed data**:
National Renewable Energy Laboratory -
Wind Integration National Dataset (WIND) Toolkit, Multi-year Annual Average

- **Solar irradiance data**:
[globalsolaratlas.info](https://globalsolaratlas.info/) Global Solar Atlas 2.0 -
Longterm yearly average of global irradiation at optimum tilt

### Energy model
A simple energy model is used to calculate the energy supply and demand for
the off-grid data centre, given the user's inputs for renewable capacities and battery storage.
On-site gas CCGT generation is assumed to kick in when no renewable generation is available.

- **Data centre demand**:
    - Data centre demand is assumed to be constant at the selected capacity
    - In reality there will be variation e.g. seasonally with ambient temperature (affecting cooling demands),
    and depending on server workload.
- **Wind generation**:
    - **Weather file:** Typical Meteorological Year (TMY) weather from [NREL](https://nsrdb.nrel.gov/data-viewer) -
    USA & Americas - Typical Meteorological Year - tmy-2024
    - Wind speed data is scaled to match the annual mean wind speed (from "Map data") at the selected hex cell
    - Power calculated using Vestas V164-8.0 power curve
- **PV generation**:
    - [pvlib-python](https://pvlib-python.readthedocs.io/en/v0.11.2/user_guide/weather_data.html)
model used to calculate Solar PV generation
    - There is no link between the solar irradiance data from "Map data" and the PV model.
    pv-lib uses it's own TMY APIs to fetch data for the location.
- **BESS generation**: Simple BESS model is used with:
    - Round trip efficiency: 95%
    - Charges only from renewable generation
    - BESS can charge to 100% State of Charge (SOC) and discharge to 0% SOC
- **Gas generation (CCGT)**:
    - Gas fills the gap after wind, PV and BESS generation
has been subtracted from demand.
    - CCGT efficiency assumed: 50%
    - CCGT is assumed to be able to provide generation at a moment's notice
    (i.e. zero cold start time). In reality 30-60 minutes may be required for a cold start.
    - Similarly, CCGT is assumed to be able to shutdown instantly.
    - CCGT capacity required is the maximum demand over the year
    (no additional capacity margin is assumed).
- **Gas fuel and CAPEX, OPEX cost assumptions for CCGT, Solar PV, BESS, Wind**:
Sourced from [GNESTE](https://github.com/iain-staffell/GNESTE),
averaged across all 2020-2025 values for each technology / cost type
(for quick development)
- **Gas emissions assumed**: 202 kgCO2 per MWh of gas consumption