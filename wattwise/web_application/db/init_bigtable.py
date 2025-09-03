from google.cloud import bigtable
from google.cloud.bigtable import column_family

PROJECT_ID = "wattwise-459502"
INSTANCE_ID = "wattwise-bigtable"
TABLE_ID = "predictions"
COLUMN_FAMILY_ID = "cf1"

client = bigtable.Client(project=PROJECT_ID, admin=True)
instance = client.instance(INSTANCE_ID)
table = instance.table(TABLE_ID)


# Tables à créer
new_tables = ["input_data", "predictions"]

for table_id in new_tables:
    table = instance.table(table_id)
    if not table.exists():
        cf1_rule = column_family.MaxVersionsGCRule(1)
        table.create(column_families={"cf1": cf1_rule})
        print(f"Table '{table_id}' created with column family 'cf1'.")
    else:
        print(f"Table '{table_id}' already exists.")
