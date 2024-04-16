import streamlit as st
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))

# Page configuration and styling
st.set_page_config(page_title="Unified Application", page_icon="🤖", layout="wide",initial_sidebar_state="collapsed")

# Importing necessary functions from other scripts
from app_functions import finsight_rag, vectordoc_hub

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Finsight RAG", "VectorDoc Hub"])

if page == "Finsight RAG":
    finsight_rag()
elif page == "VectorDoc Hub":
    vectordoc_hub()