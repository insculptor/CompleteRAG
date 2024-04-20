"""
####################################################################################
#####                       File name: app_pages.py                            #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 12/04/2024                              #####
#####                     Web Pages and Functionality                          #####
####################################################################################
"""
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from time import perf_counter as timer
from pymongo import MongoClient
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from dotenv import load_dotenv

## Conversational BufferMemory
from langchain.memory import ConversationBufferMemory
from retrieval.retrieval import get_query_embeddings, search_embedding, get_mega_chunks_by_indices, get_all_chunks_for_mega_chunks_list
from embeddings.models import get_reranked_chunks
from generation.llm import get_llm, get_input_tokens, generate_llm_response
from augmentation.prompts import create_prompt
## Chatting
from UI.htmltemplates import css, bot_template, user_template

## DocHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.models import get_device,get_embedding_model
from embeddings.vectorstore import add_to_faiss_l2_and_hnsw_indices
from embeddings.database import insert_into_mongodb

load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/FinsightRAG/.env'))


from embeddings.create_embeddings import Embeddings
create_embeddings = Embeddings(
        preprocessed_data_dir=os.environ["PREPROCESSED_DATA_DIR"],
        processed_data_dir=os.environ["PROCESSED_DATA_DIR"],
        models_base_dir=os.environ["MODELS_BASE_DIR"],
        summarization_model_name=os.environ["SUMMARIZATION_MODEL"]
    )

def finsight_rag():
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🤖 Finsight RAG </h1>", unsafe_allow_html=True)
    st.header('Retrieval-Augmented Generation Using Mega Chunks', divider="rainbow")
    # Chat template UI
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize session state for parameters
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {
            'vector_store': 'L2',
            'show_metadata': True,
            'temperature': 0.7,
            'topk': 0
        }

    def add_chat_and_generate_response(user_query):
        params = st.session_state['parameters']
        isL2Index = params['vector_store'] == 'L2'
        
        embedding_vector = get_query_embeddings(user_query)
        indices, _ = search_embedding(embedding_vector, top_n=params['topk'], isL2Index=isL2Index)
        mega_chunks_list = get_mega_chunks_by_indices(indices)
        retrieved_data, retrieved_chunks = get_all_chunks_for_mega_chunks_list(mega_chunks_list)
        reranked_chunks = get_reranked_chunks(user_query, retrieved_chunks, top_k=params['topk'])
        
        llm, tokenizer = get_llm()
        prompt = create_prompt(user_query, reranked_chunks, tokenizer)
        input_tokens = get_input_tokens(prompt, tokenizer)
        bot_response = generate_llm_response(input_tokens, llm, tokenizer)
        
        # Add user query and bot response to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        return retrieved_data, reranked_chunks

    def display_chat_history():
        for chat in st.session_state.chat_history:
            role = chat["role"]
            content = chat["content"]
            
            if role == "user":
                st.markdown(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)

    # Expander for adjusting search parameters
    with st.expander("Adjust Chat Parameters"):
        params = st.session_state.parameters
        params['vector_store'] = st.radio('Choose Vector Search Method:', ['L2', 'HNSW'], index=0, help='Select whether to use L2 or HNSW for searching.')
        params['show_metadata'] = st.checkbox('Show Metadata', value=True)
        params['temperature'] = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.05, help='Adjust the temperature to control randomness.')
        params['topk'] = st.slider('Top-k', min_value=1, max_value=10, value=5, step=1, help='Top-k Values to Sample.')

    # User query input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input("Ask a query...", key="query_input")
    with col2:
        insert_button = st.button('Ask', key='user_query')


    # Handling the user query and generating response
    if insert_button and user_query:
        retrieved_data, reranked_chunks = add_chat_and_generate_response(user_query)
        display_chat_history()

    # Conditional display of metadata and reranked chunks
    if 'show_metadata' in st.session_state.parameters and st.session_state.parameters['show_metadata']:
        try:
            if retrieved_data and reranked_chunks:
                st.subheader('Retrieved Data')
                for data in retrieved_data:
                    st.write(f"Document Name: {data.get('document_name', 'N/A')}, Page Number: {data.get('page_number', 'N/A')}")
                    st.write("Chunks:")
                    st.write(data.get('chunks', 'No chunks available'))
                    st.write("---")

                st.subheader('Reranked Chunks')
                df_reranked = pd.DataFrame(reranked_chunks)
                st.dataframe(df_reranked)
        except NameError:
            pass

    # Footer
    st.write('---')
    st.caption("RAG Playground: Explore financial insights through conversation.")


## VectorDOC Hub Page
def vectordoc_hub():
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🤖 VectorDoc Hub </h1>", unsafe_allow_html=True)
    st.header('Admin Panel to manage All Documents.', divider="rainbow")

    # Set up MongoDB connection
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_DOCUMENTS_MASTER_COLLECTION"]]
    
    # User query input
    col1, col2 = st.columns([5, 1])
    with col1:
        uploaded_files = st.file_uploader("Add Documents to Vector Hub:", accept_multiple_files=True, type='pdf')
    with col2:
        st.markdown("""
            <style>
            .stButton>button {
                color: white;  /* Changed text color to black for better visibility */
                background-color: #f63366;  /* Bright red color for the button */
                border-radius: 5px;
                border: none;  /* Removed border for a cleaner look */
                padding: 10px 20px;  /* Added padding for better button sizing */
                margin-top: 33px;  /* Ensured margin to align with other elements vertically */
                transition: background-color 0.3s ease;  /* Smooth transition for hover effect */
            }
            .stButton>button:hover {
                background-color: #cc2a49;  /* Darker shade of the button color on hover */
            }
            </style>""", unsafe_allow_html=True)
        insert_button = st.button('Insert', key='insert_docs')
    
    staging_dir = Path(os.environ.get("STAGING_DIR"))

    if insert_button:
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
                doc_data = pd.DataFrame(create_embeddings.get_document_pagewise_data(file_path))  # assuming it returns DataFrame with necessary data
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
                    all_documents_data = create_embeddings.create_chunks(doc_data)
                    llm_base_path = Path(os.environ["MODELS_BASE_DIR"])
                    llm_model_name = os.environ["SUMMARIZATION_MODEL"]
                    print(f" {llm_base_path=} | {llm_model_name=}")
                    all_documents_data = create_embeddings.add_summary_to_dataframe(all_documents_data, llm_model_name,llm_base_path,get_device())
                    angle = get_embedding_model().to(get_device())
                    all_documents_data = create_embeddings.generate_embeddings(all_documents_data, angle)
                    all_documents_data = add_to_faiss_l2_and_hnsw_indices(all_documents_data, os.environ["VECTORSTORE_BASE_DIR"])
                    cols = ['document_name', 'document_word_count', 'document_page_count','page_number', 'page_char_count', 'page_word_count','page_sentence_count', 'page_token_count', 'page_text','page_mega_chunk_count', 'mega_chunk_number', 'mega_chunk','mega_chunk_summary', 'mega_chunk_summary_embedding_index','chunks','chunks_embedding_list_index']
                    all_documents_data = all_documents_data[cols]
                    insert_into_mongodb(all_documents_data)
                    st.write(f"Successfully Inserted: {os.path.basename(file_path)}")
                    st.write(all_documents_data)
                else:
                    all_files_processed = False
                    st.write(f"Failed to process {uploaded_file.name}.")
                
                # Delete the file from staging directory after processing
                os.remove(file_path)
            
            if all_files_processed:
                st.success("All files processed and data inserted into Database successfully!")
            else:
                st.error("Some files were not processed due to errors.")
         
    # Display records from MongoDB
    st.header("Available Documents:")
    # Fetch the top 20 documents sorted by 'insert_timestamp' in descending order
    # Exclude '_id' from the results
    docs_to_display = collection.find({}, {'_id': 0}).sort('insert_timestamp', -1)
    df = pd.DataFrame(list(docs_to_display)).head(50)

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
        # Bar Chart for Document Word Count
        st.subheader('Document Word Count Overview')
        st.bar_chart(df.set_index('Document Name')['# Words'])

        # Bar Chart for Page Count
        st.subheader('Page Count Overview')
        st.bar_chart(df.set_index('Document Name')['# Pages'])
    else:
        st.write("No data found in MongoDB.")
        
    # Footer
    st.write('---')
    st.caption("RAG Playground: Explore financial insights through conversation.")
    
    
#######################
## MEGA CHUNK VIEWER ##
#######################

## Mega Chunk Viewer
def mega_chunk_viewer():
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🔍 Mega Chunk Viewer </h1>", unsafe_allow_html=True)
    st.header('Query and Display Mega Chunk Data', divider="rainbow")

    # Set up MongoDB connection
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_COLLECTION"]]

    # User input for Mega Chunk ID
    mega_chunk_id = st.text_input("Enter Mega Chunk ID:", key="mega_chunk_id")
    if st.button('Show Mega Chunk Data'):
        if mega_chunk_id:
            # Query the MongoDB collection for the entered Mega Chunk ID
            result = collection.find_one({"_id": int(mega_chunk_id)})
            if result:
                st.write("Mega Chunk Details:")
                st.json(result)  # Displaying the result as JSON for clarity
            else:
                st.write("No data found for the given Mega Chunk ID.")
        else:
            st.write("Please enter a valid Mega Chunk ID.")

    # Footer
    st.write('---')
    st.caption("Mega Chunk Viewer: Explore detailed data for specific mega chunks.")