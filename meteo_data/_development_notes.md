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


### TODO - updated 01/02/26

- Run for Canada