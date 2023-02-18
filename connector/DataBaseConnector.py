import duckdb
import os
from datetime import date
from datetime import datetime
from time import sleep

# Class to write the searched strings in database
# This should reduce the search terms whenever a term was already queried it will be fetched from the database 
# instead

class DatabaseConnector:
    # Class should initalize the database connector for a local database
    # We use duckdb here which is very fast and reliable
    # Fetching must be fast since modelling takes already quite a while
    def __init__(self, database_name) -> None:
        self.database_name: str = database_name
        self._con = None
        self.open_database()
    
    @property 
    def con(self):
        """returns the connection to the database"""
        return self._con
    
    @con.setter
    def con(self, database):
        """_summary_: Sets the database 

        Args:
            database (DuckDB): setting the database to the openend duckdb database
        """
        # check if the value is of right instance
        if isinstance(database, duckdb.DuckDBPyConnection):
            print("yes it is")
            self._con = database
          
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
    
    def open_database(self, read_only = True):
        """This creates the database
        """
        files = os.listdir()
        print(files)
    
        try:
            self._con.close()
            print("database close")
        except Exception as e:
            print("no database here")
            
            
        if self.database_name not in files:
            print("not in creating database")
            self._con = duckdb.connect(database=f"{self.database_name}", read_only=False)
            self.create_tables()
            self._con.close()
            print("database closed")
            
        self._con = duckdb.connect(database=f"{self.database_name}", read_only=read_only)
      
   
    def write_into_abstract_table(self, table):
        """_summary_: This writes the queried table into the database
        """
        if table.shape[1] != 8:
            raise ValueError("Wrong Table was supplied to the database plese check the shape: ")
        self._con.executemany("""INSERT INTO Abstract (
                                            ID, 
                                            abstract, 
                                            Title,
                                            Journal,
                                            Publication_date,
                                            first,
                                            last,
                                            search_term
                                            ) VALUES (?,?,?,?,?,?)""", table.values)

    def write_mapping_table(self, search_term):
        print("opened")
        time = date.today()
        time_now = datetime.now()
        current_time = time_now.strftime("%H:%M:%S")
        
        try:
            self._con.execute(f"INSERT INTO PubMaps VALUES ('{search_term}', '{time}', '{time_now}')")
        except Exception as e:
            print(e)
            print("the term already exists")
        
        self._con.close()
        self.open_database(read_only = True)              
        
    def show_all_tables(self):
        """This should return all tables
        returns: a dataframe pd.DataFrame holding all tables
        """
        return self._con.execute("SHOW TABLES").fetchdf()
    
    
    def create_tables(self):
        """ Creates the tables """
        mapping_table = """CREATE TABLE IF NOT EXISTS PubMaps(search_term VARCHAR PRIMARY KEY,
                            date DATE UNIQUE, time TIMESTAMP)"""

       
        self._con.execute("CREATE SEQUENCE IF NOT EXISTS unique_identifier START 1;")
        
        abstract_table = """CREATE TABLE IF NOT EXISTS Abstracts(
                            identifier integer PRIMARY KEY DEFAULT(nextval ('unique_identifier')),
                            ID VARCHAR,
                            abstract VARCHAR,
                            Title VARCHAR,
                            Journal VARCHAR,
                            Publication_date VARCHAR,
                            first VARCHAR,
                            last VARCHAR,
                            search_term VARCHAR,
                            FOREIGN KEY (search_term) REFERENCES PubMaps (search_term))"""
        
        self._con.execute(mapping_table)
        self._con.execute(abstract_table)
        print("mapped the tables")
        
    def get_mapping_table(self):
        return self._con.execute("SELECT * FROM PubMaps").fetchdf()
    
    def get_abstract_table(self):
        return self._con.execute("SELECT * FROM Abstracts").fetchdf()
    
    