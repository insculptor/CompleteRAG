import os
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv


## Conversational BufferMemory
from langchain.memory import ConversationBufferMemory
from src.retrieval.retrieval import get_query_embeddings, search_embedding, get_mega_chunks_by_indices, get_all_chunks_for_mega_chunks_list
from src.embeddings.models import get_reranked_chunks
from src.generation.llm import get_llm, get_input_tokens, generate_llm_response
from src.augmentation.prompts import create_prompt
## Chatting
from src.htmltemplates import css, bot_template, user_template

load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))

# Page configuration and styling
st.set_page_config(page_title="RAG Playground", page_icon="🤖", layout="wide")
st.markdown(css, unsafe_allow_html=True)

# Page title and introduction
st.markdown("<h1 style='text-align: center;'>🤖 Finsight RAG </h1>", unsafe_allow_html=True)
st.header('Retrieval-Augmented Generation Using Mega Chunks', divider="rainbow")

# Chat template UI
st.write('---')

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
