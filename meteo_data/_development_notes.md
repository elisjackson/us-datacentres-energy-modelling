## Development notes

Development notes for importing and processing meteo data, in preparation for use by the Plotly Dash app.

### End goals

- Display map of mean wind speed for a given year for the region
- Use Climate Data Store ED5 APIs to retrieve data
- Be able to input a country (e.g. UK), and year (e.g. 2025)
  - Country input is processed by code to create a bbox (in lat longs)
  - Import should include bbox of the country, and pull data for all hours in the time period
- Data processing
  - Imported data expected to be u, v values for a series of points within the bbox
  - Processing must calculate the mean wind speed for each point
  - Eventual file - suggest geojson, but are there better files for use in a plotly dash app?
  - May require processing to go from point data to polygon (veroni polygons?)
- Sanity checking
  - Script should be able to plot the data in a map for sanity checking the output


### Example API code for pulling ED5 data

From: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

```
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "year": ["2025"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00", "01:00"],
    "data_format": "netcdf",
    "download_format": "zip",
    "variable": [
        "100m_u_component_of_wind",
        "100m_v_component_of_wind"
    ],
    "area": [52.37, -5.45, 51.43, -2.33]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```
