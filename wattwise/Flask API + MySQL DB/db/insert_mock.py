import mysql.connector
from datetime import datetime
import csv
from dotenv import load_dotenv
import os

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)

cursor = conn.cursor()

with open("predictions.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            prediction_date = datetime.strptime(row["Date"], "%Y-%m-%d")
            electricity_demand = float(row["Electricity Demand"])
            hydro_reservoir = float(row["Hydropower reservoir"])
            hydro_ror = float(row["Hydropower run-of-river"])
            solar_pv = float(row["Solar PV Power"])
            wind_onshore = float(row["Wind Power Onshore"])
        except (ValueError, KeyError) as e:
            print(f"⚠️ Skipping row due to error: {e}")
            continue

        cursor.execute(
            """
            INSERT INTO predictions (
                prediction_date,
                electricity_demand,
                hydro_reservoir,
                hydro_run_of_river,
                solar_pv,
                wind_onshore
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (
                prediction_date,
                electricity_demand,
                hydro_reservoir,
                hydro_ror,
                solar_pv,
                wind_onshore,
            ),
        )

conn.commit()
cursor.close()
conn.close()
print("✅ Inserted predictions from CSV.")
