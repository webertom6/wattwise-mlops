import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from google.cloud.bigtable import Client
from google.cloud.bigtable.row_filters import RowKeyRegexFilter

from datetime import datetime

# Bigtable config
PROJECT_ID = "wattwise-459502"
INSTANCE_ID = "wattwise-bigtable"
PREDICTIONS_TABLE = "predictions"
INPUT_TABLE = "input_data"
COLUMN_FAMILY = "cf1"

# ----------------------------
# Function to Read Bigtable Data
# ----------------------------


@st.cache_data(ttl=300)  # Cache data for 5 minutes
def read_country_data(table_name, country_code):
    client = Client(project=PROJECT_ID, admin=True)
    instance = client.instance(INSTANCE_ID)
    table = instance.table(table_name)

    # Build regex to match rows starting with the given country code
    prefix = f"{country_code}#".encode("utf-8")
    filter_ = RowKeyRegexFilter(f"^{prefix.decode()}.*")
    rows = table.read_rows(filter_=filter_)
    rows.consume_all()

    data = []
    for row in rows.rows.values():
        row_key = row.row_key.decode("utf-8", errors="ignore")
        try:
            code, date_str = row_key.split("#", 1)
        except ValueError:
            continue  # Skip malformed keys

        row_dict = {"Date": date_str}
        for cf, columns in row.cells.items():
            for col, cells in columns.items():
                clean_col = col.decode() if isinstance(col, bytes) else str(col)
                row_dict[clean_col.strip()] = cells[0].value.decode(
                    "utf-8", errors="ignore"
                )

        data.append(row_dict)

    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(columns=["Date"])

    df.columns = [str(col).strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    return df


# ----------------------------
# Static Country Metadata
# ----------------------------
countries = pd.DataFrame(
    {
        "country": ["Spain", "France", "Germany", "Belgium", "Portugal"],
        "iso_code": ["ESP", "FRA", "DEU", "BEL", "PRT"],
        "lat": [40.4637, 46.6034, 51.1657, 50.5039, 39.3999],
        "lon": [-3.7492, 1.8883, 10.4515, 4.4699, -8.2245],
        "value": [100, 200, 150, 180, 130],
    }
)

st.set_page_config(page_title="Wattwise - Energy Dashboard", layout="wide")
st.title("Wattwise - Energy Forecast Dashboard")

# ----------------------------
# Interactive Map
# ----------------------------
fig = px.scatter_geo(
    countries,
    lat="lat",
    lon="lon",
    text="country",
    hover_name="country",
    size="value",
    size_max=20,
    projection="natural earth",
)

fig.update_traces(
    marker=dict(
        color="#2a9d8f", opacity=0.8, line=dict(width=1, color="white"), sizemode="area"
    ),
    textposition="top center",
    customdata=countries[["country"]],
    mode="markers+text",
)

fig.update_layout(
    title="Click on a country to see energy forecasts",
    title_font_size=20,
    geo=dict(
        showland=True,
        landcolor="#f5f5f5",
        showocean=True,
        oceancolor="#d0e3f1",
        bgcolor="#ffffff",
        lakecolor="#b3d9ff",
        coastlinecolor="gray",
        projection_type="natural earth",
    ),
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=600,
)

event = st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", selection_mode=["points"]
)

selected_country = None
if event and "selection" in event and event["selection"]:
    points = event["selection"].get("points", [])
    if points:
        selected_country = points[0]["customdata"][0]
        st.success(f"You selected: {selected_country}")
    else:
        st.info("Click on a country dot above.")
else:
    st.info("Click on a country dot above.")

# ----------------------------
# Data Visualization
# ----------------------------

if selected_country:
    st.header(f"Prediction Data Visualization for :blue[{selected_country}]")
    st.title("1. Energy")
    st.title("Time Series Analysis")

    # Read data from Bigtable
    country_code_map = {
        "Spain": "ES",
        "France": "FR",
        "Germany": "DE",
        "Belgium": "BE",
        "Portugal": "PT",
    }
    country_code = country_code_map.get(selected_country)

    if country_code:
        data_input = read_country_data(INPUT_TABLE, country_code)
        data_predictions = read_country_data(PREDICTIONS_TABLE, country_code)
    else:
        st.error(f"Country code not found for: {selected_country}")

    min_date_input = data_input["Date"].min()
    max_date_input = data_input["Date"].max()

    # Optional manual refresh
    if st.button("Refresh Data"):
        st.cache_data.clear()

    # Feature to display
    option_feature = st.selectbox(
        "Prediction variable to display",
        (
            "Electricity Demand",
            "Hydropower reservoir",
            "Hydropower run-of-river",
            "Solar PV Power",
            "Wind Power Onshore",
        ),
    )

    # Temporal range to display
    range_date = st.slider(
        "Range period time: ",
        min_value=datetime(1980, 1, 1),
        max_value=datetime(2025, 12, 31),
        value=(datetime(2025, 1, 1), datetime(2025, 12, 31)),
        format="YYYY - MM - DD",
    )

    # Data with range filter
    data_input_range_date = data_input[
        (data_input["Date"] >= range_date[0]) & (data_input["Date"] <= range_date[1])
    ]

    fig = go.Figure()

    option_stat = st.multiselect(
        "Which graph do you want to display: ",
        [
            "Inputs",
            "Predictions",
            "Exponential Moving Average",
            "Same month previous year",
            "Confidence interval (predictions)",
        ],
        default=["Inputs", "Predictions", "Confidence interval (predictions)"],
    )

    if "Inputs" in option_stat:
        fig.add_trace(
            go.Scatter(
                x=data_input_range_date["Date"],
                y=data_input_range_date[option_feature],
                mode="lines",
                line=dict(color="green"),
                name="Inputs",
            )
        )

    if "Predictions" in option_stat:
        fig.add_trace(
            go.Scatter(
                x=data_predictions["Date"],
                y=data_predictions[option_feature],
                mode="lines+markers",
                line=dict(color="lime"),
                name="Mean Predictions",
            )
        )

    if option_stat and "Exponential Moving Average" in option_stat:
        data_input_range_date["Ema"] = (
            data_input_range_date[option_feature].ewm(span=30).mean()
        )
        data_predictions["Ema"] = data_predictions[option_feature].ewm(span=30).mean()

        if "Inputs" in option_stat:
            fig.add_trace(
                go.Scatter(
                    x=data_input_range_date["Date"],
                    y=data_input_range_date["Ema"],
                    mode="lines",
                    line=dict(color="blueviolet"),
                    name="Exponential Moving Average of inputs",
                )
            )

        if "Predictions" in option_stat:
            fig.add_trace(
                go.Scatter(
                    x=data_predictions["Date"],
                    y=data_predictions["Ema"],
                    mode="lines",
                    line=dict(color="violet"),
                    name="Exponential Moving Average of predictions",
                )
            )

    if option_stat and "Same month previous year" in option_stat:
        option_year_comparison = st.slider(
            "Which year would you like to compare: ",
            min_value=min_date_input.year,
            max_value=max_date_input.year - 1,
            value=2024,
        )

        min_date_prediction = data_predictions["Date"].min()
        max_date_prediction = data_predictions["Date"].max()

        min_date_prediction = min_date_prediction.replace(year=option_year_comparison)
        max_date_prediction = max_date_prediction.replace(year=option_year_comparison)

        previous_input_month = data_input[
            (data_input["Date"] >= min_date_prediction)
            & (data_input["Date"] < max_date_prediction)
        ]
        previous_input_month["Shift Date"] = previous_input_month[
            "Date"
        ] + pd.DateOffset(years=max_date_input.year - option_year_comparison)

        fig.add_trace(
            go.Scatter(
                x=previous_input_month["Shift Date"],
                y=previous_input_month[option_feature],
                mode="lines",
                line=dict(color="yellow"),
                name="Same month in " + str(option_year_comparison),
            )
        )

    if "Confidence interval (predictions)" in option_stat:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([data_predictions["Date"], data_predictions["Date"][::-1]]),
                y=pd.concat(
                    [
                        data_predictions[option_feature + "_upper_bound"],
                        data_predictions[option_feature + "_lower_bound"][::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(0, 255, 128, 0.2)",
                line=dict(color="rgba(0, 255, 128, 0.1)"),
                name="95% Confidence Interval",
            )
        )

    fig.update_layout(
        title=option_feature, xaxis_title="Date", yaxis_title=option_feature
    )

    st.plotly_chart(fig)

    # Plotting
    # st.line_chart(
    #    data_input_range_date,
    #    x='Date',
    #    y = option_feature,
    #
    # )

    # option_month = st.selectbox(
    #    "Select a month: ",
    #    ("January", "February", "March", "April", "May", "Juny",
    #     "July", "Augustus", "September", "October", "November", "December")
    # )

    # months = {
    #    "January": 1,
    #    "February": 2,
    #    "March" : 3,
    #    "April" : 4,
    #    "May" : 5,
    #    "Juny": 6,
    # "July": 7,
    #  "Augustus" : 8,
    #   "September" : 9,
    #    "October" : 10,
    #     "November": 11,
    #    "December": 12
    #     }

    ## METEO FEATURES

    st.title("2. Meteo")

    st.title("Time Series Analysis")

    # Feature to display
    option_feature = st.selectbox(
        "Prediction variable to display",
        (
            "Global Horizontal Irradiance",
            "Mean Sea Level Pressure",
            "Air Temperature",
            "Total Precipitation",
            "Wind Speed_x",
            "Wind Speed_y",
        ),
    )

    # Renaming column
    # data_predictions = data_predictions.rename(columns = {option_feature: option_feature+'_prediction'})

    # Temporal range to display
    range_date_meteo = st.slider(
        "Range period time: ",
        min_value=datetime(1980, 1, 1),
        max_value=datetime(2025, 12, 31),
        value=(datetime(2025, 1, 1), datetime(2025, 12, 31)),
        format="YYYY - MM - DD",
        key="slider meteo",
    )

    # Data with range filter
    data_input_range_date_meteo = data_input[
        (data_input["Date"] >= range_date_meteo[0])
        & (data_input["Date"] <= range_date_meteo[1])
    ]

    fig = go.Figure()

    option_stat_meteo = st.multiselect(
        "Which graph do you want to display: ",
        [
            "Inputs",
            "Predictions",
            "Exponential Moving Average",
            "Same month previous year",
            "Confidence interval (predictions)",
        ],
        default=["Inputs", "Predictions", "Confidence interval (predictions)"],
        key="option stat meteo",
    )

    if "Inputs" in option_stat_meteo:
        fig.add_trace(
            go.Scatter(
                x=data_input_range_date_meteo["Date"],
                y=data_input_range_date_meteo[option_feature],
                mode="lines",
                line=dict(color="green"),
                name="Inputs",
            )
        )

    if "Predictions" in option_stat_meteo:
        fig.add_trace(
            go.Scatter(
                x=data_predictions["Date"],
                y=data_predictions[option_feature],
                mode="lines+markers",
                line=dict(color="lime"),
                name="Mean Predictions",
            )
        )

    if option_stat_meteo and "Exponential Moving Average" in option_stat_meteo:
        data_input_range_date_meteo["Ema"] = (
            data_input_range_date_meteo[option_feature].ewm(span=30).mean()
        )
        data_predictions["Ema"] = data_predictions[option_feature].ewm(span=30).mean()

        if "Inputs" in option_stat_meteo:
            fig.add_trace(
                go.Scatter(
                    x=data_input_range_date_meteo["Date"],
                    y=data_input_range_date_meteo["Ema"],
                    mode="lines",
                    line=dict(color="blueviolet"),
                    name="Exponential Moving Average of inputs",
                )
            )

        if "Predictions" in option_stat_meteo:
            fig.add_trace(
                go.Scatter(
                    x=data_predictions["Date"],
                    y=data_predictions["Ema"],
                    mode="lines",
                    line=dict(color="violet"),
                    name="Exponential Moving Average of predictions",
                )
            )

    if option_stat_meteo and "Same month previous year" in option_stat_meteo:
        option_year_comparison = st.slider(
            "Which year would you like to compare: ",
            min_value=min_date_input.year,
            max_value=max_date_input.year - 1,
            value=2024,
        )

        min_date_prediction = data_predictions["Date"].min()
        max_date_prediction = data_predictions["Date"].max()

        min_date_prediction = min_date_prediction.replace(year=option_year_comparison)
        max_date_prediction = max_date_prediction.replace(year=option_year_comparison)

        previous_input_month = data_input[
            (data_input["Date"] >= min_date_prediction)
            & (data_input["Date"] < max_date_prediction)
        ]
        previous_input_month["Shift Date"] = previous_input_month[
            "Date"
        ] + pd.DateOffset(years=max_date_input.year - option_year_comparison)

        fig.add_trace(
            go.Scatter(
                x=previous_input_month["Shift Date"],
                y=previous_input_month[option_feature],
                mode="lines",
                line=dict(color="yellow"),
                name="Same month in " + str(option_year_comparison),
            )
        )

    if "Confidence interval (predictions)" in option_stat_meteo:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([data_predictions["Date"], data_predictions["Date"][::-1]]),
                y=pd.concat(
                    [
                        data_predictions[option_feature + "_upper_bound"],
                        data_predictions[option_feature + "_lower_bound"][::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(0, 255, 128, 0.2)",
                line=dict(color="rgba(0, 255, 128, 0.1)"),
                name="95% Confidence Interval",
            )
        )

    fig.update_layout(
        title=option_feature, xaxis_title="Date", yaxis_title=option_feature
    )

    st.plotly_chart(fig)

    st.title("Input correlations")

    fig_corr = px.imshow(data_input.loc[:, data_input.columns != "Date"].corr())
    # sns.heatmap(data_input.loc[:, data_input.columns != 'Date'].corr())
    st.plotly_chart(fig_corr)
