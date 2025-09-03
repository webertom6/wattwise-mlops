from google.cloud import bigtable

client = bigtable.Client(project="wattwise-459502", admin=True)
instance = client.instance("wattwise-bigtable")

old_tables = ["input_data", "predictions"]
for table_id in old_tables:
    table = instance.table(table_id)
    if table.exists():
        table.delete()
        print(f"Table '{table_id}' deleted.")
    else:
        print(f"Table '{table_id}' not found.")
