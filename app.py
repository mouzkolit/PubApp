import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
import os

# set the config for the webpage
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide")

@st.cache
def load_data():
    """ Function to load data, later connect to AWS"""
    current_path = os.getcwd()
    impact_factor = pd.read_csv(current_path + "/data/impact_factor.txt", delimiter = "\t")
    print(impact_factor)
    return impact_factor
    
def main():
    #start paradigm of the APP and First Visualizations
    st.sidebar.title("Data-Science in Neuroscience: Beginner's Guide")


if __name__ == "__main__":
    impact_factor = load_data()
    main()
    