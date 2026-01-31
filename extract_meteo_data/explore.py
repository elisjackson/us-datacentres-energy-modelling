from owslib.csw import CatalogueServiceWeb
import wind_power_timeseries as tm

# csw = CatalogueServiceWeb(
#     "https://csw.s-enda.k8s.met.no",
#     version="2.0.2"
# )

# Download from server and save to files
wind_data = tm.download.retrieve_nora3(
    windfarms,time_start,time_end,use_cache=True,data_path=data_path)

csw = CatalogueServiceWeb(
    "https://csw.s-enda.k8s.met.no",
)

csw.getrecords2(
    keywords=["MEPS"],
    maxrecords=10
)

csw.getrecords2(
    constraints=["MEPS"],
    maxrecords=10
)

csw.getrecords2(
    constraints=["wind", "atmosphere"],
    maxrecords=20
)

print(len(csw.records))
