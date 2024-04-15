import streamlit as st
import os
from pathlib import Path
import streamlit as st
import unicodedata
from time import perf_counter as timer
from pymongo import MongoClient
from tqdm.auto import tqdm
import torch
import tempfile
import shutil
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.embeddings.create_embeddings import get_all_documents_data,create_chunks,add_summary_to_dataframe,generate_embeddings,get_document_pagewise_data
from src.embeddings.models import get_device,get_embedding_model
from src.embeddings.vectorstore import add_to_faiss_l2_and_hnsw_indices
from src.embeddings.database import insert_into_mongodb,update_chunk_mapping_collection,insert_into_document_master
from dotenv import load_dotenv
# Load environment variables
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))
from src.htmltemplates import css

# Set up MongoDB connection
client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
db = client[os.environ["MONGODB_DB"]]
collection = db[os.environ["MONGODB_DOCUMENTS_MASTER_COLLECTION"]]


def main():
    # Page configuration and styling
    st.set_page_config(page_title="🤖 VectorDoc Hub", page_icon="🤖", layout="wide")
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🤖 VectorDoc Hub </h1>", unsafe_allow_html=True)
    st.header('Admin Panel to manage All Documents.', divider="rainbow")    
    # File uploader allows user to upload multiple PDFs
    uploaded_files = st.file_uploader("Add Documents to Vector Hub:", accept_multiple_files=True, type='pdf')
    
    # Set up MongoDB connection
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_DOCUMENTS_MASTER_COLLECTION"]]
    staging_dir = Path(os.environ.get("STAGING_DIR"))
    if uploaded_files:
        # Ensure the staging directory exists
        if not os.path.exists(staging_dir):
            os.makedirs(staging_dir)

        all_files_processed = True
        for uploaded_file in uploaded_files:
            # Save each uploaded file to the staging directory
            file_path = os.path.join(staging_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Check if the document already exists in MongoDB
            doc_data = pd.DataFrame(get_document_pagewise_data(file_path))  # assuming it returns DataFrame with necessary data
            if not doc_data.empty:
                doc_name = os.path.splitext(uploaded_file.name)[0]
                doc_word_count = int(doc_data['document_word_count'].iloc[0])
                doc_page_count = int(doc_data['document_page_count'].iloc[0])
                # Search for existing documents by name, word count, and page count
                existing_doc = collection.find_one({'document_name': doc_name,
                                                    'document_word_count': doc_word_count,
                                                    'document_page_count': doc_page_count})
                if existing_doc:
                    st.write(f"{uploaded_file.name} already exists in the database.")
                    continue  # Skip this file
                else:
                    # Check if a document with the same name exists but different content
                    if collection.find_one({'document_name': doc_name}):
                        # Add a version suffix to document name if only name matches
                        version = 2
                        while collection.find_one({'document_name': f"{doc_name}_v{version}"}):
                            version += 1
                        doc_name = f"{doc_name}_v{version}"
                        doc_data['document_name'] = doc_name

                # Process and insert the new/updated version of the document
                all_documents_data = create_chunks(doc_data)
                llm_base_path = Path(os.environ["MODELS_BASE_DIR"])
                llm_model_name = os.environ["SUMMARIZATION_MODEL"]
                print(f" {llm_base_path=} | {llm_model_name=}")
                add_summary_to_dataframe(all_documents_data, llm_model_name,llm_base_path,get_device())
                angle = get_embedding_model().to(get_device())
                all_documents_data = generate_embeddings(all_documents_data, angle)
                all_documents_data = add_to_faiss_l2_and_hnsw_indices(all_documents_data, os.environ["VECTORSTORE_BASE_DIR"])
                cols = ['document_name', 'document_word_count', 'document_page_count','page_number', 'page_char_count', 'page_word_count','page_sentence_count', 'page_token_count', 'page_text','page_mega_chunk_count', 'mega_chunk_number', 'mega_chunk','mega_chunk_summary', 'mega_chunk_summary_embedding_index','chunks','chunks_embedding_list_index']
                all_documents_data = all_documents_data[cols]
                insert_into_mongodb(all_documents_data)
                st.write(f"Metadata generated for: {file_path}")
                st.write(all_documents_data)

                #insert_into_document_master(embeddings)
            else:
                all_files_processed = False
                st.write(f"Failed to process {uploaded_file.name}.")
            
            # Delete the file from staging directory after processing
            os.remove(file_path)
        
        if all_files_processed:
            st.success("All files processed and data inserted into MongoDB successfully!")
        else:
            st.error("Some files were not processed due to errors.")
         
    # Display records from MongoDB
    st.header("Available Documents:")
    # Fetch the top 20 documents sorted by 'insert_timestamp' in descending order
    # Exclude '_id' from the results
    docs_to_display = collection.find({}, {'_id': 0}).sort('insert_timestamp', -1)
    df = pd.DataFrame(list(docs_to_display)).head(20)

    if not df.empty:
        # Rename column names for better readability in the UI
        df = df.rename(columns={
            'document_name': 'Document Name',
            'document_word_count': '# Words',
            'document_page_count': '# Pages',
            'insert_timestamp': 'Insert Timestamp'
        })

        # Format the 'Insert Timestamp' for better readability, if the column exists
        if 'Insert Timestamp' in df.columns:
            df['Insert Timestamp'] = pd.to_datetime(df['Insert Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Display the DataFrame using Streamlit
        st.dataframe(df)
    else:
        st.write("No data found in MongoDB.")
if __name__ == "__main__":
    main()
