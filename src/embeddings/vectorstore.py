"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/25/2024                              #####
#####                   Create and Manage FAISS Vectorstore                    #####
####################################################################################
"""

## Load Environment Variables
import os
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/FinsightRAG/.env'))



def _load_or_create_hnsw_faiss_index(vectorstore_base_path, dim=1024, M=32, efConstruction=200):
    """Load or create a FAISS HNSW index."""
    index_path = os.path.join(vectorstore_base_path, os.environ["FAISS_HNSW_INDEX"])
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = efConstruction
    return index


def _load_or_create_l2_faiss_index(vectorstore_base_path, dim=1024):
    """Load or create a FAISS index."""
    index_path = os.path.join(vectorstore_base_path, os.environ["FAISS_L2_INDEX"])
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        # Using FlatL2 index for simplicity; choose other types as needed
        index = faiss.IndexFlatL2(dim)
    return index

def _save_embedding_to_faiss(index, embedding):
    """Save a single embedding to FAISS index and return its index."""
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    index.add(embedding)
    return index.ntotal - 1

def _save_embeddings_list_to_faiss(index, embeddings_list):
    """Save a list of embeddings to FAISS index and return their indices."""
    indices = []
    for embedding in embeddings_list:
        idx = _save_embedding_to_faiss(index, embedding)
        indices.append(idx)
    return indices


def add_faiss_indices_to_dataframe(df, vectorstore_base_path):
    """Add FAISS L2 index columns for embeddings to the dataframe."""
    index = _load_or_create_l2_faiss_index(vectorstore_base_path)
    df['mega_chunk_summary_embedding_index'] = df['mega_chunk_summary_embedding'].apply(lambda x: _save_embedding_to_faiss(index, x))
    df['chunks_embedding_list_index'] = df['chunks_embedding_list'].apply(lambda x: _save_embeddings_list_to_faiss(index, x))
    faiss.write_index(index, os.path.join(vectorstore_base_path, os.environ["FAISS_L2_INDEX"]))
    return df

def add_to_faiss_l2_and_hnsw_indices(df, vectorstore_base_path, dim=1024):
    """Add embeddings to both L2 and HNSW FAISS indices and update dataframe with indices."""
    l2_index = _load_or_create_l2_faiss_index(vectorstore_base_path, dim)
    hnsw_index = _load_or_create_hnsw_faiss_index(vectorstore_base_path, dim)
    
    # Process 'mega_chunk_summary_embedding'
    indices = []
    for embedding in tqdm(df['mega_chunk_summary_embedding'], desc="Adding mega chunk embeddings"):
        l2_idx = _save_embedding_to_faiss(l2_index, embedding)
        _save_embedding_to_faiss(hnsw_index, embedding)  # Assuming HNSW index uses the same order
        indices.append(l2_idx)
    df['mega_chunk_summary_embedding_index'] = indices
    
    # Process 'chunks_embedding_list'
    chunk_indices = []
    for embeddings_list in tqdm(df['chunks_embedding_list'], desc="Adding chunk list embeddings"):
        chunk_list_indices = _save_embeddings_list_to_faiss(l2_index, embeddings_list)
        # No need to save again for HNSW as the order is maintained
        chunk_indices.append(chunk_list_indices)
    df['chunks_embedding_list_index'] = chunk_indices
    
    faiss.write_index(l2_index, os.path.join(vectorstore_base_path, os.environ["FAISS_L2_INDEX"]))
    faiss.write_index(hnsw_index, os.path.join(vectorstore_base_path, os.environ["FAISS_HNSW_INDEX"]))
    
    return df