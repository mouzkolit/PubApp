import streamlit as st
from connector.DataBaseConnector import DatabaseConnector
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer


def whole_database():
    """_summary_: This should be the main function for the whole database analysis
    """
    database = DatabaseConnector(True)
    st.header("Explore the your PubMed Database")
    st.markdown("---")
    # check if something was selected
    if selected := st.sidebar.multiselect(
        "Select already searched data:",
        database.con.execute("SELECT * FROM PubMedKeys")
        .fetchdf()["key"]
        .values):

        collected_dataframe = pd.DataFrame()
        for key in selected:
            key = key.replace(" ", "_")
            inter_df = database.con.execute(f"Select * FROM {key}").fetch_df()
            inter_df["cluster"] = key + " " + inter_df["labels"].astype(str)
            collected_dataframe = pd.concat([collected_dataframe, inter_df], axis = 0)

        select_cluster = st.sidebar.selectbox("Select Cluster:", collected_dataframe["cluster"].unique())
        edited_cells = build_collect_dataframe(collected_dataframe)
        st.markdown("---")

        show_abstracts_pubmed(edited_cells, collected_dataframe)
        st.markdown("---")
        if select_cluster:
            generate_wordloud_from_df(collected_dataframe, select_cluster)


# TODO Rename this here and in `whole_database`
def generate_wordloud_from_df(collected_dataframe: pd.DataFrame, select_cluster: str) -> None:
    """_summary_: This should generate the wordcloud from the selected cluster

    Args:
        collected_dataframe (pd.DataFrame): The dataframe with the collected data
        select_cluster (str): The selected cluster
    """
    st.subheader("Word Cloud of the selected Cluster:")
    wordcloud = generate_wordcloud(collected_dataframe[collected_dataframe["cluster"] == select_cluster])
    fig,ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.write(fig)


def build_collect_dataframe(collected_dataframe: pd.DataFrame) -> dict:
    """_summary_: This should build the dataframe for the collected data

    Args:
        collected_dataframe (pd.DataFrame): The dataframe which should be displayed
    Returns:
        dict: Session State of the data editor which should retrieve if a cells is edited or not
    """
    st.subheader("Selected Publications:")
    collected_dataframe["Show Abstract"] = False
    columns = ["Show Abstract", "ID", "Title", "Journal", "Publication_date", "first", "last"]
    st.experimental_data_editor(collected_dataframe[columns], key="data_editor")

    return st.session_state["data_editor"]["edited_cells"]

def show_abstracts_pubmed(edited_cells: dict, collected_dataframe: pd.DataFrame) -> None:
    """_summary_: This should show the abstracts of the selected publications
    in an expander

    Args:
        edited_cells (dict): The session state of the data editor with selected pubs
        collected_dataframe (pd.DataFrame): The dataframe with the collected abstract data
    """
    st.subheader("Abstracts of the selected Publications:")
    st.info("You can select the publications you want to save by clicking on the checkbox in the table below.")
    col1, col2 = st.columns(2)
    counter = 0
    for index, value in edited_cells.items():
        if value:
            df = collected_dataframe.iloc[int(index.split(":")[0])][["ID","Title","abstract"]]
            with st.expander(df["Title"]):
                st.markdown("The Link to the Publication: https://pubmed.ncbi.nlm.nih.gov/" + str(df["ID"]) + "/")
                st.write(df["abstract"])
        counter += 1

@st.cache_data
def generate_wordcloud(collected_dataframe: pd.DataFrame):
    """_summary_: This should generate a word cloud of the selected cluster

    Args:
        collected_dataframe (pd.DataFrame): The dataframe with all cluster and the processed abstracts
    """
    print("Generate Word Cloud")
    collected_dataframe["string_words"] = [" ".join(i) for i in collected_dataframe["processed_abstracts"]]

    vectorizer = TfidfVectorizer(stop_words='english')
    vecs = vectorizer.fit_transform(collected_dataframe["string_words"])
    feature_names = vectorizer.get_feature_names_out()
    dense = vecs.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    df = df.T.sum(axis=1)
    wordcloud = WordCloud(background_color="white", max_words=20).generate_from_frequencies(df)
    return wordcloud


if __name__ == "__main__":
    whole_database()