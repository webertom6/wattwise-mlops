import os
import zipfile
import cdsapi


import config

dataset = "sis-energy-derived-reanalysis"

request_meteo = {
    "variable": [
        "wind_speed_at_100m",
        "wind_speed_at_10m",
        "surface_downwelling_shortwave_radiation",
        "pressure_at_sea_level",
        "2m_air_temperature",
        "total_precipitation",
    ],
    "spatial_aggregation": ["country_level"],
    "temporal_aggregation": ["daily"],
}

request_energy = {
    "variable": [
        "electricity_demand",
        "hydro_power_generation_reservoirs",
        "hydro_power_generation_rivers",
        "solar_photovoltaic_power_generation",
        "wind_power_generation_onshore",
    ],
    "spatial_aggregation": ["country_level"],
    "energy_product_type": ["power"],
    "temporal_aggregation": ["daily"],
}

for key, request in zip(["meteo", "energy"], [request_meteo, request_energy]):
    print(f"Requesting {key} data from CDS...")
    client = cdsapi.Client(url=config.CDS_URL, key=config.CDS_KEY)
    target = f"notebooks/data_prep/{key}2.zip"
    client.retrieve(dataset, request).download(target=target)

    print(f"Downloaded {key} data from CDS.")

    print(f"Unzipping {key} data...")

    with zipfile.ZipFile(target, "r") as zip_ref:
        zip_ref.extractall(target.replace(".zip", ""))

    print(f"Unzipped {key} data.")

    print(f"Cleaning up {key} data...")

    os.remove(target)

    print(f"Cleaned up {key} data.")

    print(f"Finished requesting {key} data from CDS.")
