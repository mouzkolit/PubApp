# PubApp

## Streamlit App fetching PubMed Abstracts from a queried Search String

* Multiprocessing abstract fetch together with Title/Date/Author/Last-Author/PMID from PubMed.org
* Preprocessing of the Abstract, removing stopwords, punctuation, lower-case and perform tokeniztion
* Training of a Doc2Vec model for Document Embeddings as well as training a Berttopic Model
* UMAP Latent-Space Visualization as well as DB-scan/Louvain clustering to retrieve putative Topics
* Topic Modelling about the Topics per Cluster using Bert-TOPIC
* Saves the searched data into a database that is querable for the search key (DuckDB)
* At the moment it should be used locally since DuckDB has no concurrent writing possibilites

## What is missing
 
* Author Network Integration Citations Impact of Citations (Currently in development)
* Summary of Authors Work (This will be provided by means of searching an author directly)
* Summary of search query Term (This can be used whenever LLMS are more affordable)
* DuckDB DataBase that can be used for a local copy of searched data

# EDA and UMAP

Exploratory Data Analysis will be performed and topics extracted from the Abstracts as well as that PubMed Abstracts will be clustered

<img src="/Images/eda_example.png" title="hover text">
<br>
<img src="/Images/umap_example.png" title="hover text">



