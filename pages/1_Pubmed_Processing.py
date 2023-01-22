import streamlit as st
import pandas as pd
import re 
import os
from gensim.parsing.preprocessing import preprocess_string,strip_tags, strip_punctuation, remove_stopwords
from gensim.models import Phrases
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora, models
import umap
import plotly.express as px
from sklearn.neighbors import kneighbors_graph
import leidenalg as la
import igraph
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns
from bertopic import BERTopic
from connector.PubMedConnector import PubMedConnector


@st.cache
def load_data():
    """ Function to load data, later connect to AWS"""
    current_path = os.getcwd()
    return pd.read_csv(current_path + "/data/impact_factor.txt", delimiter = "\t")

# make a list of stopwords
def processing_data(impact_factor):
    """_summary_:  Get the impact factor list 

    Args:
        impact_factor (pd.DataFrame): DataFrame of impact factors
    """
    # layout of the streamlit page that is necessary for the training
    st.sidebar.subheader("Choose your search term")
    if url := st.sidebar.text_input('Enter your search query here:'):
        
        stopword_list = ["among","although","especially","kg","km","mainly","ml","mm",
                        "disease","significantly","obtained","mutation","significant",
                        "quite","result","results","estimated","interesting","conducted",
                        "associated","performed","respectively","larger","genes","gene",
                        "mutations","related","expression","pattern","mutation","clc","identified",
                        "suprisingly","preferentially","subsequently","far","little","known","importantly",
                        "synonymous","skipping","father","mother","pedigree","novo","rescues","rescued","restored",
                        "exhibits","induce", "Background","Objective","Methods","cells", "kinase","activation","protein"]

        abstract_data, tab3 = retrieve_abstract_data(url, stopword_list)
        bert_topic_modelling(abstract_data,tab3)

def retrieve_abstract_data(url, stopword_list):
    """_summary_

    Args:
        url (_type_): _description_
        stopword_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    st.header("PubMed Abstract Processing:")
    pubmed_fetch = PubMedConnector(url)
    result = pubmed_fetch.query_data()
    return process_abstract_data(result,stopword_list)
    
    
def process_abstract_data(result, stopword_list):
    """_summary_

    Args:
        result (_type_): _description_
        stopword_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    result["date"] = [int(i[:4]) for i in result["Publication_date"].tolist()]
    tab1, tab2, tab3 = st.tabs(["Overview","Abstract Umap","Content Analysis"])

    # make columns for the tabs

    tab1_col1, tab1_col2 = tab1.columns(2)

        #tab1 is the overview tab
    count_occurences_per_year(result, tab1_col1)
    count_occurences_per_journal(result, tab1_col2)
    top_10_first_authors(result, tab1_col1)
    top_10_last_authors(result, tab1_col2)
            

    #preprocessing of the abstracts that are searched for, remove punctuations, stopwords
    result["processed_abstracts"] = preprocessing_list(result["abstract"], stopword_list)
    bigram = Phrases(result["processed_abstracts"].tolist(),min_count = 5, threshold = 100)
    document = [TaggedDocument(doc, [i]) for i, doc in zip(result["ID"].tolist(),bigram[result["processed_abstracts"].tolist()])]
    with st.spinner("Word Vector Model is running..."):
        model = running_model(document)
        st.success("Model done")

    if model:
        df_umap = umap_visualization(model)
        df_umap["Title"] = result["Title"].tolist()
        df_umap["Journal"] = result["Journal"].tolist()
        umap_figure = draw_umap(df_umap)
        tab2.plotly_chart(umap_figure, use_container_width = True)

    result["labels"] = df_umap["labels"]

    return result, tab3
            
def preprocessing_list(liste,stopword_list):
    """ preprocess the abstract list, remove stopwords, punctuations, numbers
    input: 
    liste: list of abstracts
    stopword_list: list of stopwords
    returns"""
    processed_abstracts = []

    for i in liste:
        CUSTOM_FILTERS = [lambda x: x.lower(),remove_stopwords]
        a = preprocess_string(i, CUSTOM_FILTERS)
        no_integers = [x for x in a if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        no_integers = [x for x in no_integers if x not in stopword_list]
        no_integers = [re.sub("[^A-Za-z0-9|-]","",x) for x in no_integers]
        processed_abstracts.append(no_integers)
    return processed_abstracts


@st.cache
def running_model(document):
    """ run the model that detect similarities using the Doc2Vec model
    here we can implement other models such as Bert and more too!
    input:
    document: list of abstracts
    returns:
    model: trained model"""
    model_stream = models.Doc2Vec(
        document,
        window = 15,
        min_count=4,
        epochs = 5,
        workers = 8,
        dm=0, 
        dbow_words=1)
    model_stream.train(document,total_examples=model_stream.corpus_count,epochs=model_stream.epochs)
    return model_stream

def umap_visualization(model):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    doc_tags = model.dv.vectors 
    X = doc_tags
    
    prediction_labels = umap.UMAP(n_neighbors=5,
                                    min_dist=0.0,
                                    n_components=2,
                                    random_state=42,
                                    metric = "cosine"
                                    ).fit_transform(X)

    df_dx = pd.DataFrame(prediction_labels, columns=['UMAP-1', 'UMAP-2'])
    # make a neighborhood graph
    neighborhood_graph = kneighbors_graph(prediction_labels, 10, mode='connectivity', include_self=True)
    neighborhood_graph = neighborhood_graph.toarray()
    g = igraph.Graph.Adjacency((neighborhood_graph > 0).tolist())
    partition = la.find_partition(g,la.ModularityVertexPartition).membership
    df_dx["labels"] = partition
    return df_dx

def draw_umap(umap_df):
    """ draw the umap visualization
    input:
    umap_df: dataframe with the umap coordinates
    returns:
    umap_figure: figure of the umap visualization
    """
    fig3 = px.scatter(umap_df, 
                      x="UMAP-1", 
                      y="UMAP-2", 
                      color="labels", 
                      hover_data=["Title", "Journal"],
                      template="plotly_white", 
                      title = "UMAP visualization of queried term")

    return fig3

def count_occurences_per_year(abstract_df,tab):
    """ retrieves the number of occurences per year
    input:
    abstract_df: dataframe of abstracts
    tab: streamlit tab column
    """
    grouped_occurences = abstract_df.groupby("date")["date"].count()
    year_plot = px.bar(grouped_occurences, x=grouped_occurences.index, y=grouped_occurences.values)
    tab.plotly_chart(year_plot, use_container_width = True)

def count_occurences_per_journal(abstract_df,tab):
    """_summary_

    Args:
        abstract_df (_type_): _description_
        tab (_type_): _description_
    """
    grouped_occurences = abstract_df.groupby("Journal")["Journal"].count().sort_values(ascending = False).iloc[:10]
    journal_plot = px.bar(grouped_occurences, 
                          x=grouped_occurences.index, 
                          y=grouped_occurences.values,
                          )
    tab.plotly_chart(journal_plot, use_container_width = True)

def top_10_first_authors(abstract_df,tab):
    """_summary_

    Args:
        abstract_df (_type_): _description_
        tab (_type_): _description_
    """
    grouped_occurences = abstract_df.groupby("first")["first"].count().sort_values(ascending = False).iloc[:10]
    first_plot = px.bar(grouped_occurences,
                        x=grouped_occurences.index,
                        y=grouped_occurences.values,
                        title="Top 10 first authors")
    tab.plotly_chart(first_plot, use_container_width = True)

def top_10_last_authors(abstract_df,tab):
    """_summary_

    Args:
        abstract_df (_type_): _description_
        tab (_type_): _description_
    """
    grouped_occurences = abstract_df.groupby("last")["last"].count().sort_values(ascending = False).iloc[:10]
    last_plot = px.bar(grouped_occurences, 
                       x=grouped_occurences.index, 
                       y=grouped_occurences.values, 
                       title = "Top 10 last authors")
    tab.plotly_chart(last_plot, plot = True)

@st.cache()
def bert_topic_modelling(df_end, tab):
    """_summary_: Calculates topics analysis using BertTopics

    Args:
        df_end (pd.DataFrame): DataFrame holding the processed abstracts
        tab (st.Tabs): Tab where to draw the Analysis
    """
    print(df_end.head())
    df_end["string_abstract"] = [" ".join(i) for i in df_end["processed_abstracts"]]
    targets = df_end["labels"].tolist()
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(df_end["string_abstract"].tolist())
    topics_per_class = topic_model.topics_per_class(df_end["string_abstract"].tolist(), classes=targets)
    tab.plotly_chart(topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10))
   
    
def write_table_model(model_list):
    """

    Args:
        model_list (_type_): _description_
    """
    
    pass
        
    
if __name__ == "__main__":
    impact_factor = load_data()
    processing_data(impact_factor)
