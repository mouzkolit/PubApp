import streamlit as st
from connector.DataBaseConnector import DatabaseConnector
import pandas as pd
import numpy as np
import seaborn as sns


def whole_database():
    """_summary_: This should be the main function for the whole database analysis
    """
    database = DatabaseConnector(True)
    trial_table = database.con.execute("SELECT * FROM PubMedKeys").fetchdf()
    st.sidebar.multiselect("Select already searched data:", trial_table["key"].values)
    st.dataframe(trial_table)

if __name__ == "__main__":
    whole_database()