import streamlit as st
from connector.DataBaseConnector import DatabaseConnector

def load_database():
    database = DatabaseConnector("pubmed.db")
    database.open_database()
    return database

def main(database):
    """_summary_: This allows to have a look into the tables of the Database
    Furthermore it should be used for downloading the specific tables of interes

    Args:
        database (duck.DuckDBConnections): Database connection
    """
    col1, col2 = st.columns(2)
    mapping_table = database.get_mapping_table()
    abstract_table = database.get_abstract_table()
    col1.write(mapping_table)
    col2.write(abstract_table)

if __name__ == "__main__":
    database = load_database()
    main(database)
