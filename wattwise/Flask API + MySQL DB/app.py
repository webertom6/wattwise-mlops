from flask import Flask, jsonify, render_template
import mysql.connector
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
app = Flask(__name__)


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
    )


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/application")
def application():
    energy_sources = [
        "Electricity Demand",
        "Hydropower reservoir",
        "Hydropower run-of-river",
        "Solar PV Power",
        "Wind Power Onshore",
    ]
    return render_template("application.html", energy_sources=energy_sources)


@app.route("/country/<country>")
def country_page(country):
    energy_sources = [
        "Electricity Demand",
        "Hydropower reservoir",
        "Hydropower run-of-river",
        "Solar PV Power",
        "Wind Power Onshore",
    ]
    return render_template(
        "country.html", country=country, energy_sources=energy_sources
    )


@app.route("/predict/<country>/<energy>", methods=["GET"])
def predict(country, energy):
    conn = get_db_connection()
    cursor = conn.cursor()

    column_map = {
        "Electricity Demand": "electricity_demand",
        "Hydropower reservoir": "hydro_reservoir",
        "Hydropower run-of-river": "hydro_run_of_river",
        "Solar PV Power": "solar_pv",
        "Wind Power Onshore": "wind_onshore",
    }

    column = column_map.get(energy)
    if column is None:
        return jsonify({"error": f"Invalid energy source: {energy}"}), 400

    try:
        cursor.execute(
            f"""
            SELECT prediction_date, {column}
            FROM predictions
            WHERE {column} IS NOT NULL
            ORDER BY prediction_date
        """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.strftime("%b %d")

        stats = {
            "min": df["value"].min(),
            "max": df["value"].max(),
            "avg": round(df["value"].mean(), 2),
        }

        return render_template(
            "forecast.html",
            country=country,
            energy=energy,
            data=df.to_dict(orient="records"),
            stats=stats,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predictions", methods=["GET"])
def get_predictions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT prediction_date, electricity_demand, hydro_reservoir, hydro_run_of_river, solar_pv, wind_onshore
        FROM predictions
        ORDER BY prediction_date DESC
    """
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    columns = [
        "date",
        "electricity_demand",
        "hydro_reservoir",
        "hydro_run_of_river",
        "solar_pv",
        "wind_onshore",
    ]
    return jsonify([dict(zip(columns, row)) for row in rows])


@app.route("/remove", methods=["GET"])
def remove_predictions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "All predictions removed."})


if __name__ == "__main__":
    app.run(debug=True)
