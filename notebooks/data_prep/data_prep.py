import os

import pandas as pd


def open_df_title_unit_csv(file_path):
    # Read the file and store comments
    with open(file_path, "r") as file:
        lines = file.readlines()

    comments = [line for line in lines if line.startswith("#")]
    data = [line for line in lines if not line.startswith("#")]

    # # Extract the title and unit from comments
    title_index = comments.index(
        next(
            line
            for line in comments
            if line.startswith("## Title") or line.startswith("## File content")
        )
    )
    title = comments[title_index + 1].strip()
    title = title.replace("#", "").strip()
    # Extract the unit from comments if there is formated like this:
    """
    ## Unit
    ### MWh
    """
    unit_index = comments.index(
        next(line for line in comments if line.startswith("## Unit"))
    )
    unit = comments[unit_index + 1].strip()
    unit = unit.replace("#", "").strip()

    # Write the data back to a temporary file
    temp_file_path = file_path + ".tmp"
    with open(temp_file_path, "w") as temp_file:
        temp_file.writelines(data)

    # Display the comments
    # for comment in comments:
    #     print(comment.strip())

    # Read the data with pandas
    df = pd.read_csv(temp_file_path, index_col="Date")

    # Optionally, remove the temporary file
    os.remove(temp_file_path)

    return df, title, unit


super_directory = r".//data"
directory1 = r".//data/Meteorology"
directory2 = r".//data/Energy"


def prepare_data(super_directory, directory1, directory2, save=False):
    df_es = pd.DataFrame()

    for directory in os.listdir(super_directory):
        print(f"Processing directory: {directory}")
        directory_path = os.path.join(super_directory, directory)
        print(f"Processing directory: {directory_path}")
        # Loop through each file in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                try:
                    file_path = os.path.join(directory_path, filename)
                    print(f"Reading file: {file_path}")

                    df, title, unit = open_df_title_unit_csv(file_path)
                    print(f"Title: {title}")
                    title_parts = title.split()
                    if len(title_parts) > 5:
                        title = " ".join(title_parts[:2])
                    print(f"Title after processing: {title}")
                    print(f"Unit: {unit}")

                    # Ensure the dataset contains an 'ES' column to filter for Spain
                    if "ES" in df.columns:
                        # Keep only the 'Date' and 'ES' columns
                        df = df[["ES"]].copy()
                        df.reset_index(inplace=True)
                    else:
                        print(f"Skipping {filename} as no 'ES' column found.")
                        continue

                    # Convert 'Date' column to datetime format before merging
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

                    # Drop rows where 'Date' could not be converted (if any)
                    df = df.dropna(subset=["Date"])

                    # Rename the 'ES' column with the title name
                    df.rename(columns={"ES": title}, inplace=True)

                    # Merge into main DataFrame (on 'Date')
                    if df_es.empty:
                        df_es = df  # First dataset initializes df_es
                    else:
                        df_es = pd.merge(
                            df_es, df, on="Date", how="outer"
                        )  # Merge on 'Date'

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # Filter data to keep only records from 2020 onwards
    df_es = df_es[df_es["Date"] >= "1980-01-01"]

    # Display rows where date conversion failed (if any remain after dropping NaNs)
    invalid_dates = df_es[df_es["Date"].isna()]
    if not invalid_dates.empty:
        print("Warning: Some date values could not be converted:")
        print(invalid_dates)

    if save:
        # Save the final DataFrame to a CSV file
        df_es.to_csv("notebooks/data_prep/spain_energy_meteo_data.csv", index=False)

    return df_es


def check_df_reference(df_es):
    # Load the reference DataFrame
    reference_file_path = "./notebooks/data_prep/ref_spain_energy_meteo_data.csv"
    df_reference = pd.read_csv(reference_file_path)

    ismatching = True

    # Check if the columns match
    if list(df_es.columns) != list(df_reference.columns):
        print(
            "Warning: The columns of the created DataFrame do not match the reference DataFrame."
        )
        print("Created DataFrame columns:", df_es.columns)
        print("Reference DataFrame columns:", df_reference.columns)
        ismatching = False
    else:
        print("The columns of the created DataFrame match the reference DataFrame.")

    # Check if the starting date matches
    created_start_date = df_es["Date"].min()
    reference_start_date = pd.to_datetime(df_reference["Date"]).min()

    if created_start_date != reference_start_date:
        print(
            "Warning: The starting date of the created DataFrame does not match the reference DataFrame."
        )
        print("Created DataFrame starting date:", created_start_date)
        print("Reference DataFrame starting date:", reference_start_date)
        ismatching = False
    else:
        print(
            "The starting date of the created DataFrame matches the reference DataFrame."
        )

    return ismatching


if __name__ == "__main__":
    # Prepare the data
    df_es = prepare_data(super_directory, directory1, directory2, save=False)

    # Check if the DataFrame matches the reference
    ismatching = check_df_reference(df_es)

    if ismatching:
        print("The created DataFrame matches the reference DataFrame.")
    else:
        print("The created DataFrame does not match the reference DataFrame.")
