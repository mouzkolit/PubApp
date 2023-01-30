import duckdb
import os

# Class to write the searched strings in database
# This should reduce the search terms
#

class DatabaseConnector:
    
    def __init__(self) -> None:
        self.database_name: str = "PubMedDataBase.duckdb"
        self.con = None
        if not self.check_database():
            self.create_database()
        self.write_database()
            
    def write_database(self):
        
        """This should write the database with the key (pubmed search key)
        and the acquired data
        
        DataBase holds a table with keys and the associated table name
        in addition tables with the data will be created
        """
        pass
    
    def check_database(self) -> bool:
        """_summary_: This should check if the database is already in the path

        Returns:
            bool: _description_
        """
        return self.database_name in os.listdir()
    
    def create_database(self):
        """This creates the database
        """
        self.con = duckdb.connect(database=self.database_name, read_only=False)
        
    
    def write_table_into_database(self):
        """_summary_
        """
        q = "CREATE TopicsMap (ID INT, SearchTerm VARCHAR, Time TIMESTAMP, date DATE UNIQUE)"
        self.con.execute(q)
        
        
    def show_all_tables(self):
        """Tnhis should return all tables

        Returns:
            list: _description_
        """
        return self.con.execute("SHOW TABLES").fetchall()