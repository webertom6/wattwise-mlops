import pandas as pd
from google.cloud import bigtable

# Configuration
PROJECT_ID = "wattwise-459502"
INSTANCE_ID = "wattwise-bigtable"
COLUMN_FAMILY = "cf1"


# Charger les CSV
predictions_meteo_df = pd.read_csv("predictions_meteo_interval.csv")
predictions_energy_df = pd.read_csv("predictions_energy_interval.csv")
input_data_df = pd.read_csv("spain_energy_meteo_data.csv")
client = bigtable.Client(project="wattwise-459502", admin=True)
instance = client.instance("wattwise-bigtable")


def write_df_to_bigtable(df, table_id, country_code="ES"):
    table = instance.table(table_id)
    if not table.exists():
        print(f"Table '{table_id}' does not exist.")
        return

    for _, row in df.iterrows():
        date_str = str(row["Date"])[:10]  # assume format: YYYY-MM-DD
        row_key = f"{country_code}#{date_str}"
        bt_row = table.direct_row(row_key.encode())

        for col in df.columns:
            if col == "Date":
                continue
            value = str(row[col])
            bt_row.set_cell("cf1", col, value)

        bt_row.commit()


write_df_to_bigtable(predictions_energy_df, table_id="predictions", country_code="ES")
write_df_to_bigtable(predictions_meteo_df, table_id="predictions", country_code="ES")
write_df_to_bigtable(input_data_df, table_id="input_data", country_code="ES")
