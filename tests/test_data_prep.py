import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from notebooks.data_prep.data_prep import open_df_title_unit_csv


def test_open_df_title_unit_csv():
    # Test the function with a sample file
    file_path = r"./tests/test_data/H_ERA5_ECMW_T639_EDM_NA---_Euro_NUT0_S197901010000_E202501312300_NRG_TIM_01d_NA-_noc_org_NA_NA---_NA---_GamNT.csv"
    df, title, unit = open_df_title_unit_csv(file_path)

    # Check if the DataFrame is not empty
    assert not df.empty

    # Check if the title and unit are correct
    assert title == "Electricity Demand (EDM), expressed as Energy (NRG)"
    assert unit == "MWh"
