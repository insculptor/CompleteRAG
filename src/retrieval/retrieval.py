"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/27/2024                              #####
#####             Retrieve Data from Vectorstore based on User Query           #####
####################################################################################
"""

## Load Environment Variables
import os
from pathlib import Path
from time import perf_counter as timer
from tqdm.auto import tqdm

from pymongo import MongoClient
import faiss
import numpy as np
from angle_emb import AnglE
from transformers import pipeline
import torch
import json

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,BitsAndBytesConfig






import torch
import fitz
import pandas as pd
from dotenv import load_dotenv
from embeddings.models import get_embedding_model
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/FinsightRAG/.env'))



def get_query_embeddings(query):
    llm_base_path =  Path(os.environ["MODELS_BASE_DIR"])
    embedding_model_name = os.environ["EMBEDDING_MODEL"]
    angle =get_embedding_model(llm_base_path,embedding_model_name)
    query_embedding = angle.encode(query, to_numpy=True)
    print(f"[INFO]: Generated Embedding of shape: {query_embedding.shape}")
    return query_embedding


def search_embedding(embedding_vector, top_n=5,isL2Index=True):
    """
    Searches the FAISS index for the top N closest embeddings to the given embedding vector.

    Parameters:
    - faiss_index (faiss.Index): The pre-built and trained FAISS index.
    - embedding_vector (numpy.array): The query embedding vector, shape should be (1, D) where D is the dimension of vectors in the index.
    - top_n (int): The number of nearest neighbors to return.

    Returns:
    - (distances, indices) (tuple): The distances and indices of the top N closest embeddings in the index.
    """
    vectorstore_base_path = os.environ["VECTORSTORE_BASE_DIR"]
    
    if isL2Index:
        faiss_index_path = os.path.join(vectorstore_base_path, os.environ["FAISS_L2_INDEX"])
    else:
        faiss_index_path = os.path.join(vectorstore_base_path, os.environ["FAISS_HNSW_INDEX"])
    faiss_index = faiss.read_index(faiss_index_path)
    # Ensure the embedding_vector is a numpy array and has the right shape (1, dimension)
    embedding_vector = np.array(embedding_vector).reshape(1, -1).astype('float32')
    
    # Perform the search
    distances, indices = faiss_index.search(embedding_vector, top_n)
    
    return [int(x) for x in list(indices.flatten())] , distances.flatten()

def get_mega_chunks_by_indices(index_list):
    """
    Retrieves documents from a MongoDB collection by a list of indices.
    
    Parameters:
    - index_list (list): A list of indices to retrieve documents for.
    
    Returns:
    - list: A list of documents retrieved from the MongoDB collection.
    """
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_MAPPING_COLLECTION"]]

    # Create a query to retrieve documents where '_id' matches any of the indices in the list
    query = {'_id': {'$in': index_list}}
    
    # Specify the projection to return only the 'mega_chunk_summary_embedding_index' field
    projection = {'_id': 0, 'mega_chunk_summary_embedding_index': 1}
    
    # Execute the query and retrieve the documents
    documents = list(collection.find(query, projection))

    # Extract 'mega_chunk_summary_embedding_index' from each document
    mega_chunk_summary_embedding_indices = [doc['mega_chunk_summary_embedding_index'] for doc in documents]
    
    return list(set(mega_chunk_summary_embedding_indices))


def get_all_chunks_for_mega_chunks_list(index_list):
    """
    Retrieves documents from a MongoDB collection by a list of indices.
    
    Parameters:
    - index_list (list): A list of indices to retrieve documents for.
    
    Returns:
    - list: A list of documents retrieved from the MongoDB collection.
    """
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_COLLECTION"]]

    # Create a query to retrieve documents where '_id' matches any of the indices in the list
    query = {'_id': {'$in': index_list}}
    
    # Specify the projection to return only the 'mega_chunk_summary_embedding_index' field
    projection = {'_id': 0, 'document_name': 1,
                  'document_word_count':2,'document_page_count':3,'page_char_count':4,'page_word_count':5, 'page_number':6,
                  'page_sentence_count':7,'page_token_count':8,'page_text':9,'mega_chunk':10, 'mega_chunk_summary':11,
                  'chunks':12,'mega_chunk_summary_embedding_index':13,'chunks_embedding_list_index':14}

    
    
    # Execute the query and retrieve the documents
    documents = list(collection.find(query, projection))

    # Extract 'mega_chunk_summary_embedding_index' from each document
    chunks = [item for sublist in [doc['chunks'] for doc in documents] for item in sublist]
    
    return documents,list(set(chunks))