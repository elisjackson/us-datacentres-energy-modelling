import xarray as xr

f = r"C:\Users\Elis\Downloads\e2cb02e8bfdf70dd09f6a16de94de22e\data_stream-oper_stepType-instant.nc"

ds = xr.open_dataset(f)
print(ds)

u = ds["u100"]
v = ds["v100"]