import xarray as xr

ds = xr.open_dataset("raw/2021/pressure/data.nc")

df = ds.to_dataframe().reset_index()

print(df)
