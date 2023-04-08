import streamlit as st
import pandas as pd
import re
import os
import openai
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from gensim.models import Phrases
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora, models
import umap
import plotly.express as px
from sklearn.neighbors import kneighbors_graph
import leidenalg as la
import igraph
from wordcloud import WordCloud

from bertopic import BERTopic
from connector.PubMedConnector import PubMedConnector
from connector.DataBaseConnector import DatabaseConnector
from summarizer import Summarizer


@st.cache_data
def load_data():
    """ Function to load data, later connect to AWS"""
    current_path = os.getcwd()
    return pd.read_csv(current_path + "/data/impact_factor.txt", delimiter = "\t")

# make a list of stopwords
def processing_data():
    """_summary_:  Get the impact factor list

    Args:
        impact_factor (pd.DataFrame): DataFrame of impact factors
    """
    # layout of the streamlit page that is necessary for the training
    st.sidebar.subheader("Choose your search term")
    st.info("Please enter a your PubMed search-term!")
    if url := st.sidebar.text_input('Enter your search query here:'):
        database = DatabaseConnector(False) # False means that the database is not created
        stopword_list = ["among","although","especially","kg","km","mainly","ml","mm",
                        "disease","significantly","obtained","mutation","significant",
                        "quite","result","results","estimated","interesting","conducted",
                        "associated","performed","respectively","larger","genes","gene",
                        "mutations","related","expression","pattern","mutation","clc","identified",
                        "suprisingly","preferentially","subsequently","far","little","known","importantly",
                        "synonymous","skipping","father","mother","pedigree","novo","rescues","rescued","restored",
                        "exhibits","induce", "Background","Objective","Methods","cells", "kinase","activation","protein"]
        try:
            abstract_data, tab3, df_umap = retrieve_abstract_data(url, stopword_list)
            if url not in database.con.execute("SELECT key FROM PubMedKeys").fetchall():
                st.write("This key is already in the database")
                database.write_database(abstract_data, df_umap, url, url)
            bert_topic_modelling(abstract_data,tab3)

        except Exception as e:
            print(e)
        finally:
            # This is important in case of a failure and therefore database is not close which kind of disrupts the app
            database.con.close()

def retrieve_abstract_data(url, stopword_list):
    """_summary: Retrieves available abstract data from Pubmed
    using multiprocessing and saves this into a result data frame

    Args:
        url (str): Search Term in Pubmed
        stopword_list (list): Additional Stopwords fitting to PubMed Abstracts

    Returns:
        pd.DataFrame: Returns DataFrame holding abstracts, title and more per chunk
    """
    st.header("PubMed Abstract Processing:")
    pubmed_fetch = PubMedConnector(url)
    result = pubmed_fetch.query_data()
    return process_abstract_data(result,stopword_list)

def process_abstract_data(result: pd.DataFrame, stopword_list: list):
    """_summary_: Process the abstract data and creaet exploratory analysis
    from the top 10 authors and journals

    Args:
        result (pd.DataFrame): This includes the abstracts, title, authors, journal, publication date
        stopword_list (_type_): _description_

    Returns:
        tuple: result table, the current tab and the umap table
    """

    result["date"] = [int(i[:4]) for i in result["Publication_date"].tolist()]
    tab1, tab2, tab3, tab4 = st.tabs(["Overview","Abstract Umap","Content Analysis", "Summarizer"])

    # make columns for the tabs

    tab1_col1, tab1_col2 = tab1.columns(2)

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

    # check if the model was successfully generated
    if model:
        df_umap = create_umap_visualization_from_model(model, result, tab2)
            #summarize_bert_text(df_umap, tab4)

    result["labels"] = df_umap["labels"]
    return result, tab3, df_umap


def create_umap_visualization_from_model(model, result: pd.DataFrame, tab2):
    """_summary_: Create the umap visualization from the model

    Args:
        model (Doc2VEc): Doc2Vec Model
        result (pd.DataFrame): _description_
        tab2 (st.Tabs): Tab to display the umap visualization

    Returns:
        pd.DataFrame: _description_
    """
    result_umap = umap_visualization(model)
    result_umap["Title"] = result["Title"].tolist()
    result_umap["Journal"] = result["Journal"].tolist()
    result_umap["abstract"] = result["abstract"].tolist()
    tab2.slider("Select the cluster", 0,1)
    umap_figure = draw_umap(result_umap)
    tab2.plotly_chart(umap_figure, use_container_width = True)
    return result_umap

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

@st.cache_resource
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
    if df_dx.shape[0] < 100:
        k = 10
        l = 0.1
    elif df_dx.shape[0] < 1000:
        k = 20
        l = 0.01
    elif df_dx.shape[0] > 1000:
        k = 30
        l = 0.001
    else:
        st.warning("The number of points is to low")
        return None

    neighborhood_graph = kneighbors_graph(prediction_labels,
                                          k,
                                          mode='connectivity',
                                          include_self=True,
                                          n_jobs = -1)

    neighborhood_graph = neighborhood_graph.toarray()
    g = igraph.Graph.Adjacency((neighborhood_graph > 0).tolist())
    partition = la.CPMVertexPartition(g, resolution_parameter = l)
    optimiser = la.Optimiser()
    diff = optimiser.optimise_partition(partition)
    df_dx["labels"] = partition.membership
    print(df_dx)
    return df_dx

def draw_umap(umap_df):
    """ draw the umap visualization
    input:
    umap_df: dataframe with the umap coordinates
    returns:
    umap_figure: figure of the umap visualization
    """
    return px.scatter(
        umap_df,
        x="UMAP-1",
        y="UMAP-2",
        color="labels",
        hover_data=["Title", "Journal"],
        template="plotly_white",
        title="UMAP visualization of queried term",
    )

def count_occurences_per_year(abstract_df,tab):
    """ retrieves the number of occurences per year
    input:
    abstract_df: dataframe of abstracts
    tab: streamlit tab column
    """
    grouped_occurences = abstract_df.groupby("date")["date"].count()
    year_plot = px.bar(grouped_occurences,
                       x=grouped_occurences.index,
                       y=grouped_occurences.values,
                       title = "# Publication per Year")
    tab.plotly_chart(year_plot, use_container_width = True)

def count_occurences_per_journal(abstract_df,tab):
    """_summary_: Retrieves the aggregated counts per journal

    Args:
        abstract_df (pd.DataFrame): the dataframe hodling all the abstracts
        tab (st.col): streamlit tab column
    """
    grouped_occurences = abstract_df.groupby("Journal")["Journal"].count().sort_values(ascending = False).iloc[:10]
    journal_plot = px.bar(grouped_occurences,
                          x=grouped_occurences.index,
                          y=grouped_occurences.values,
                          title = "# Publication per Journal"
                          )
    tab.plotly_chart(journal_plot, use_container_width = True)

def top_10_first_authors(abstract_df,tab):
    """_summary_: Retrieves the top 10 first authors from the abstract

    Args:
        abstract_df (pd.DataFrame): dataframe holding all the abstract
        tab (st.col): streamlit tab column
    """
    grouped_occurences = abstract_df.groupby("first")["first"].count().sort_values(ascending = False).iloc[:10]
    first_plot = px.bar(grouped_occurences,
                        x=grouped_occurences.index,
                        y=grouped_occurences.values,
                        title="Top 10 first authors")
    tab.plotly_chart(first_plot, use_container_width = True)

def top_10_last_authors(abstract_df,tab):
    """_summary_: retrieves the top10 last authors

    Args:
        abstract_df (pd.DataFrame): _description_
        tab (st.col): _description_
    """
    grouped_occurences = abstract_df.groupby("last")["last"].count().sort_values(ascending = False).iloc[:10]
    last_plot = px.bar(grouped_occurences,
                       x=grouped_occurences.index,
                       y=grouped_occurences.values,
                       title = "Top 10 last authors")
    tab.plotly_chart(last_plot, use_container_width = True)


def bert_topic_modelling(df_end, tab):
    """_summary_: Calculates topics analysis using BertTopics

    Args:
        df_end (pd.DataFrame): DataFrame holding the processed abstracts
        tab (st.Tabs): Tab where to draw the Analysis
    """
    print(df_end.head())
    df_end["string_abstract"] = [" ".join(i) for i in df_end["processed_abstracts"]]
    targets = df_end["labels"].tolist()
    topic_model = BERTopic(nr_topics="auto")
    topics, probs = topic_model.fit_transform(df_end["string_abstract"].tolist())
    topics_per_class = topic_model.topics_per_class(df_end["string_abstract"].tolist(), classes=targets)

    #col1, col2 = tab.columns(2)
    #col1.plotly_chart(topic_model.visualize_topics(), use_container_width = True)
    tab.plotly_chart(topic_model.visualize_topics_per_class(topics_per_class), use_container_width = True)


def summarize_ai_text(results, tab):
    """_summary_: This can summarize the text using the OPEN AI gpt model

    Args:
        results (pd.DataFrame): _description_
        tab (st.tab): _description_
    """
    summaries_dict = {}
    for i in results["labels"].unique():
        df_text = results[results["labels"]==i]
        print(df_text.head())
        text = ".".join(df_text["abstract"].tolist()) # should be the label text
        augmented_prompt = f"Summarize this text: {text}"
        summary = openai.Completion.create(
                            model="ada",
                            prompt=augmented_prompt,
                            temperature=.5,
                            max_tokens=1000,
                                )
        summaries_dict[i].append(summary)
        break

def summarize_bert_text(results, tab):
    """_summary_: This can summarize the text using the Bert Models

    Args:
        results (pd.DataFrame): Holding the Abstract Data
        tab (st.Tabs): _description_
    """
    summaries_dict = {} # should hold the summaries
    model= Summarizer() # creates the summarizer model
    for i in results["labels"].unique():
        df_text = results[results["labels"]==i]
        text = ".".join(df_text["abstract"].tolist()) # should be the label text
        summary = model(text, ratio = 0.05)
        print(summary)
        summaries_dict[i].append(summary)
        break


if __name__ == "__main__":
    impact_factor = load_data()
    processing_data()

