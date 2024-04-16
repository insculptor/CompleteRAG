"""
####################################################################################
#####                          File name: app.py                               #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 15/04/2024                              #####
#####                Streamlit Application UI (run this)                       #####
####################################################################################
"""
import os
import streamlit as st
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
# Load environment variables
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))

# Page configuration and styling
st.set_page_config(page_title="Finsight RAG", page_icon="🤖", layout="wide",initial_sidebar_state="collapsed")

# Importing necessary functions from other scripts
from src.UI.app_pages import finsight_rag, vectordoc_hub,mega_chunk_viewer

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Finsight RAG", "VectorDoc Hub", "Mega Chunks" ])

if page == "Finsight RAG":
    finsight_rag()
elif page == "VectorDoc Hub":
    vectordoc_hub()
elif page == "Mega Chunks":
    mega_chunk_viewer()