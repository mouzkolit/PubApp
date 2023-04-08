import streamlit as st
from connector.DataBaseConnector import DatabaseConnector
import pandas as pd
import numpy as np
import seaborn as sns


def whole_database():
    """_summary_: This should be the main function for the whole database analysis
    """
    database = DatabaseConnector(True)
    # check if something was selected
    if selected := st.sidebar.multiselect(
        "Select already searched data:",
        database.con.execute("SELECT * FROM PubMedKeys")
        .fetchdf()["key"]
        .values,
    ):
        collected_dataframe = pd.DataFrame()
        for key in selected:
            key = key.replace(" ", "_")
            inter_df = database.con.execute(f"Select * FROM {key}").fetch_df()
            print(inter_df)
            collected_dataframe = pd.concat([collected_dataframe, inter_df], axis = 0)

        edited_cells = build_collect_dataframe(collected_dataframe)
        show_abstracts_pubmed(edited_cells, collected_dataframe)
        # lets take the edited cells and make an expander for this


def build_collect_dataframe(collected_dataframe: pd.DataFrame) -> dict:
    """_summary_: This should build the dataframe for the collected data

    Args:
        collected_dataframe (pd.DataFrame): The dataframe which should be displayed
    Returns:
        dict: Session State of the data editor which should retrieve if a cells is edited or not
    """
    st.subheader("Selected Publications:")
    collected_dataframe["Show Abstract"] = False
    columns = ["Show Abstract", "ID", "Title", "Journal", "Publication_date", "first", "last"]
    st.experimental_data_editor(collected_dataframe[columns], key="data_editor")

    return st.session_state["data_editor"]["edited_cells"]

def show_abstracts_pubmed(edited_cells: dict, collected_dataframe: pd.DataFrame) -> None:
    """_summary_: This should show the abstracts of the selected publications
    in an expander

    Args:
        edited_cells (dict): The session state of the data editor with selected pubs
        collected_dataframe (pd.DataFrame): The dataframe with the collected abstract data
    """
    st.subheader("Abstracts of the selected Publications:")
    col1, col2 = st.columns(2)
    counter = 0
    for index, value in edited_cells.items():
        if counter % 2 == 0:
            col = col1
        else:
            col = col2
        if value:
            df = collected_dataframe.iloc[int(index.split(":")[0])][["ID","Title","abstract"]]
            with col.expander(df["Title"]):
                col.markdown("The Link to the Publication: https://pubmed.ncbi.nlm.nih.gov/" + str(df["ID"]) + "/")
                col.write(df["abstract"])
        counter += 1


if __name__ == "__main__":
    whole_database()