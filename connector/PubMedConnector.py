from Bio import Entrez
from Bio import Medline
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from itertools import islice
from urllib.error import HTTPError
import pandas as pd
import streamlit as st


class PubMedConnector:
    def __init__(self,url):
        
        self.email = "maximilian.zeidler@i-med.ac.at"
        self.url = url
        self.count = 0
        self.pubmed_data = None
                
    def query_data(self):
        """_summary_: Should query the distinct url string in Pubmed
        """
        try:
            return self.query_pubmed()
        except HTTPError:
            count += 1
            time.sleep(10)
            if count < 5:
                self.query_data(self.url, self.count)
            else:
                st.error("No connection to the server could be established")
                raise ConnectionError

    # TODO Rename this here and in `query_data`
    def query_pubmed(self):
        Entrez.mail = self.email
        esearch_query = Entrez.esearch(db="pubmed", term=self.url,retmax = 200000)
        esearch_result = Entrez.read(esearch_query)
        count = esearch_result['Count']
        esearch_query1 = Entrez.esearch(db="pubmed", term=self.url, retmax = count, retmode = "xlm")
        esearch_result1 = Entrez.read(esearch_query1)
        idlist = esearch_result1["IdList"]
        handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text", retmax = count)
        records = Medline.parse(handle)
        return self.retrieve_data_from_record(records)
            
    def retrieve_data_from_record(self,records):
        """
        doc_function detects pubmed articles
        input: search query
        output: dictionary of:
        abstracts, titles, journals, PMIDS
        """

        # this can be abstracted to a dictionary
        # try block to retrieve the data for the query
        chunked_records = self.chunk(list(records), 50)
        records_dataframe_list = self.progress_records_multiprocessing(chunked_records)
        return self.append_dataframes(records_dataframe_list)
        
       
    def progress_records_multiprocessing(self,records_list, workers = 8):
        """Runs the Worker Thread for each chunk and returns an iterator of the list
        results of the records fetched from the PubMed Outline
        Args:
            new_list (_type_): _description_
        Returns:
            _type_: _description_
        """
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return executor.map(self.retrieve_records, records_list, timeout=60)
        
    def retrieve_records(self,records):
        """_summary_

        Args:
            records (_type_): _description_

        Returns:
            _type_: _description_
        """
        corpus_dictionary = {"ID": [], 
                            "abstract": [],
                            "Title":[], 
                            "Journal": [],
                            "Publication_date":[], 
                            "first":[], 
                            "last": []
                            }
      
        for record in records:
            if (("DP") in record.keys() and ("PMID") in record.keys() and ("AB") in record.keys() and ("TI") in record.keys() and ("JT") in record.keys() and ("FAU") in record.keys()):
                self.create_dictionary_from_record(corpus_dictionary, record)
            else:
                print("Searched record is not available")
           
        return pd.DataFrame(corpus_dictionary)

    # TODO Rename this here and in `retrieve_records`
    def create_dictionary_from_record(self, corpus_dictionary, record):
        """_summary_

        Args:
            corpus_dictionary (_type_): _description_
            record (_type_): _description_
        """
        corpus_dictionary["ID"].append(record["PMID"])
        corpus_dictionary["abstract"].append(record["AB"])
        corpus_dictionary["Title"].append(record["TI"])
        corpus_dictionary["Journal"].append(record["JT"])
        corpus_dictionary["Publication_date"].append(record["DP"])
        corpus_dictionary["first"].append(record["FAU"][0])
        corpus_dictionary["last"].append(record["FAU"][-1])

    def chunk(self,it, size):
        """List will be chunked into equal size for request Since request size is only 1000
        Args:
            it (list): will be the iterable of the list
            size (int): describes the number of chunks based on size per chunk
        Returns:
            iter : iterable object for all chunks 
        """
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
    
    def append_dataframes(self,records):
        """_summary_

        Args:
            records (_type_): _description_

        Returns:
            _type_: _description_
        """
        final_records = pd.DataFrame()
        for i in records:
            final_records = pd.concat([final_records, i], axis = 0)
        return final_records
