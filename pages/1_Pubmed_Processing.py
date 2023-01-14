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

@st.cache
def load_data():
    """ Function to load data, later connect to AWS"""
    current_path = os.getcwd()
    impact_factor = pd.read_csv(current_path + "/data/impact_factor.txt", delimiter = "\t")
    return impact_factor

# make a list of stopwords
def processing_data(impact_factor):
    """_summary_

    Args:
        impact_factor (_type_): _description_
    """
    # layout of the streamlit page that is necessary for the training
    
    st.header("Welcome to Pubmed Analysis")
    stopword_list = ["among","although","especially","kg","km","mainly","ml","mm",
                    "disease","significantly","obtained","mutation","significant",
                    "quite","result","results","estimated","interesting","conducted",
                    "associated","performed","respectively","larger","genes","gene",
                    "mutations","related","expression","pattern","mutation","clc","identified",
                    "suprisingly","preferentially","subsequently","far","little","known","importantly",
                    "synonymous","skipping","father","mother","pedigree","novo","rescues","rescued","restored",
                    "exhibits","induce", "Background","Objective","Methods","cells", "kinase","activation","protein"]

    st.sidebar.subheader("Choose your search term")
    url = st.sidebar.text_input('Enter your search query here:')

    # check if search term was entered
    if url:
        st.header("PubMed Abstract Processing:")
        abstract_data = doc_function(url) # get the abstracts from pubmed
        print(abstract_data.head())
        
        abstract_data["date"] = [int(i[:4]) for i in abstract_data["Publication_date"].tolist()]
        tab1, tab2, tab3 = st.tabs(["Overview","Abstract Umap","Content Analysis"])

        # make columns for the tabs

        tab1_col1, tab1_col2 = tab1.columns(2)

        #tab1 is the overview tab
        count_occurences_per_year(abstract_data,tab1_col1)
        count_occurences_per_journal(abstract_data,tab1_col2)
        top_10_first_authors(abstract_data,tab1_col1)
        top_10_last_authors(abstract_data,tab1_col2)
        

        #preprocessing of the abstracts that are searched for, remove punctuations, stopwords
        end_list = preprocessing_list(abstract_data["abstract"],stopword_list)
        bigram = Phrases(end_list,min_count = 5, threshold = 100)
        document = [TaggedDocument(doc, [i]) for i, doc in zip(defined_lists[2],bigram[end_list])]
        with st.spinner("Word Vector Model is running..."):
            model = running_model(document)
            st.success("Model done")

        if model:
            df_umap = umap_visualization(model)
            df_umap["Title"] = abstract_data["Title"]
            df_umap["Journal"] = abstract_data["Journal"]
            umap_figure = draw_umap(df_umap)
            tab2.write(umap_figure)
        
        abstract_data["labels"] = df_umap["labels"]

    lda_topic_modelling(abstract_data, 6, stopword_list)
            
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
    trans = umap.UMAP(n_neighbors=100,metric = "cosine", random_state=42, min_dist = 0.1).fit_transform(X)
    prediction_labels = umap.UMAP(n_neighbors=30,
                                    min_dist=0.0,
                                    n_components=2,
                                    random_state=42,
                                    metric = "cosine"
                                    ).fit_transform(X)

    df_dx = pd.DataFrame(prediction_labels, columns=['UMAP-1', 'UMAP-2'])


    # make a neighborhood graph
    neighborhood_graph = kneighbors_graph(prediction_labels, 100, mode='connectivity', include_self=True)
    neighborhood_graph = neighborhood_graph.toarray()
    g = igraph.Graph.Adjacency((neighborhood_graph > 0).tolist())
    partition = la.find_partition(g, la.CPMVertexPartition,resolution_parameter = 0.005).membership
    df_dx["labels"] = partition
    
    return df_dx

def draw_umap(umap_df):
    """ draw the umap visualization
    input:
    umap_df: dataframe with the umap coordinates
    returns:
    umap_figure: figure of the umap visualization
    """
    fig3 = px.scatter(umap_df, x="UMAP-1", y="UMAP-2", color="labels", hover_data=["Title", "Journal"],
                    template="plotly_white", title = "UMAP visualization of queried term")

    fig3.write_html("umap_msk.html")
    return fig3

def count_occurences_per_year(abstract_df,tab):
    """ retrieves the number of occurences per year
    input:
    abstract_df: dataframe of abstracts
    tab: streamlit tab column
    """
    grouped_occurences = abstract_df.groupby("date")["date"].count()
    year_plot = px.bar(grouped_occurences, x=grouped_occurences.index, y=grouped_occurences.values)
    tab.write(year_plot)

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
    tab.write(journal_plot)

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
    tab.write(first_plot)

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
    tab.write(last_plot)

@st.cache()
def lda_topic_modelling(df_end, cluster_number,stopword_list):
    """_summary_

    Args:
        df_end (_type_): _description_
        cluster_number (_type_): _description_
        stopword_list (_type_): _description_
    """
    model_list = []
    df_end["abstract"] = preprocessing_list(df_end["abstract"].tolist(),stopword_list)

    for i in df_end["labels"].unique():
        df_labels = df_end[df_end["labels"]== i]
        id2word = corpora.Dictionary(df_labels["abstract"].values)
        corpus = [id2word.doc2bow(text) for text in df_labels["abstract"].values]
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]
        ldamodel = models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=1,
                                                alpha='auto',
                                                eta='auto',
                                                iterations=800,
                                                passes = 20,
                                                eval_every=1)

        model_list.append(ldamodel)


    word_cloud_lda(model_list, stopword_list)

def word_cloud_lda(model_topics, stopwords):
    """_summary_

    Args:
        model_topics (_type_): _description_
        stopwords (_type_): _description_
    """
    for index, models in enumerate(model_topics):
        fig, ax = plt.subplots()
        topics = models.show_topics(formatted=False)
        cloud = WordCloud(stopwords=stopwords,
                    background_color='white',
                    width=800,
                    height=800,
                    max_words=30,
                    colormap='tab10',
                    prefer_horizontal=1.0)

        topic_word = dict(topics[0][1])
        cloud.generate_from_frequencies(topic_word, max_font_size=150)
        sns.despine()
        plt.imshow(cloud)
        plt.xlabel("Vector Frequency Space 1")
        plt.ylabel("Vector Frequency Space 2")

        plt.savefig(f"model_{index}_word_cloud.pdf", transparent = True, dpi=500, bbox_inches = "tight")
        
    #st.write(fig)


if __name__ == "__main__":
    impact_factor = load_data()
    processing_data(impact_factor)
