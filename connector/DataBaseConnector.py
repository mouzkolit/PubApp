import duckdb
import os
import pandas as pd

# Class to write the searched strings in database
# This should reduce the search terms
#

class DatabaseConnector:
    # This is the Database connector singleton
    # This should be used to write the data to the database
    # This should be used to read the data from the database
    # This should be used to check if the data is already in the database
    # This should be used to check if the database is already in the path
    def __init__(self, read_only: bool = False) -> None:
        self.database_name: str = "PubMedDataBase.duckdb"
        self.read_only: bool = read_only
        self.con = None
        if not self.check_database():
            self.create_database()
        else:
            self.con = duckdb.connect(database=self.database_name, read_only=self.read_only)

    def write_database(self, table: pd.DataFrame, umap_table: pd.DataFrame, table_name: str, search_key: str):

        """This should write the database with the key (pubmed search key)
        and the acquired data

        DataBase holds a table with keys and the associated table name
        in addition tables with the data will be created
        """
        try:
            table_df = table #.to_sql(table_name, self.con, if_exists="replace")
            table_name = "_".join(table_name.split(" ")) # create a table name without spaces
            um_table = f"umap_{table_name}"
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM table_df")
            self.con.execute(f"CREATE TABLE {um_table} AS SELECT * FROM umap_table")
            self.con.execute("INSERT INTO PubMedKeys (key, table_name, umap_table) VALUES (?, ?, ?)", (search_key, table_name, um_table))
        except Exception as e:
            print(e)
        finally:
            self.con.close() # close the connection

    def check_database(self) -> bool:
        """_summary_: This should check if the database is already in the path

        Returns:
            bool: Returns if database is found in the path or not
        """
        return self.database_name in os.listdir()

    def create_database(self):
        """This creates the database
        """
        self.con = duckdb.connect(database=self.database_name, read_only=False)
        create_unique_offline_analysis_sequence = """CREATE SEQUENCE ID;"""
        self.con.execute(create_unique_offline_analysis_sequence)
        self.con.execute("""CREATE TABLE PubMedKeys(ID_Pub integer PRIMARY KEY DEFAULT(nextval ('ID')),
                                                    key VARCHAR,
                                                    table_name VARCHAR,
                                                    umap_table VARCHAR)""")

