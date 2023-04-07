# PubApp

## Streamlit App fetching PubMed Abstracts from a queried Search String

* Multiprocessing abstract fetch together with Title/Date/Author/Last-Author/PMID
* Preprocessing of the Abstract, removing stopwords, punctuation, lower-case and more
* Training of a Doc2Vec model for Document Embeddings
* UMAP Latent-Space Visualization as well as DB-scan/Louvain clustering to retrieve putative Topics
* Topic Modelling about the Topics per Cluster using Bert-TOPIC


## What is missing
 
* currently a database will be setted up that will held already queried documents (Amazon DynamoDB)
* Author Network Integration Citations Impact of Citations
* Summary of Authors Work
* Summary of search query Term

# EDA and UMAP

Exploratory Data Analysis will be performed and topics extracted from the Abstracts as well as that PubMed Abstracts will be clustered

<img src="/Images/eda_example.png" title="hover text">
<br>
<img src="/Images/umap_example.png" title="hover text">



