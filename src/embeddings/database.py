"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/25/2024                              #####
#####              Create and Manage MongoDB CRUD Operations                   #####
####################################################################################
"""


## Load Environment Variables
import os
from pathlib import Path
import faiss
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))

def insert_into_mongodb(df):
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_COLLECTION"]]
    df['_id'] = df['mega_chunk_summary_embedding_index']
    data_dict = df.to_dict("records")
    collection.insert_many(data_dict)
    update_chunk_mapping_collection(df)
    insert_into_document_master(df)
    print(f"[INFO]: Successfully Inserted!")
    
def update_chunk_mapping_collection(input_df):
    """
    Transforms the DataFrame according to the specified operations for MongoDB insertion.
    
    Parameters:
    - input_df (pd.DataFrame): The input DataFrame with columns 
      'mega_chunk_summary_embedding_index' and 'chunks_embedding_list_index'.
    
    Returns:
    - pd.DataFrame: Transformed DataFrame with self mapping and '_id' column.
    """
    input_df = input_df[['chunks_embedding_list_index','mega_chunk_summary_embedding_index']]
    # Explode the 'chunks_embedding_list' column
    exploded_df = input_df.explode('chunks_embedding_list_index')
    
    # Add self-mapping rows by copying 'mega_chunk_summary_embedding' into a new row with the same value for '_id'
    self_map_df = pd.DataFrame({
        'mega_chunk_summary_embedding_index': input_df['mega_chunk_summary_embedding_index'],
        'chunks_embedding_list_index': input_df['mega_chunk_summary_embedding_index']
    })
    
    # Rename 'chunks_embedding_list' to '_id' in both DataFrames
    exploded_df = exploded_df.rename(columns={'chunks_embedding_list_index': '_id'})
    self_map_df = self_map_df.rename(columns={'chunks_embedding_list_index': '_id'})
    
    # Concatenate the exploded DataFrame with the self-mapping DataFrame
    final_df = pd.concat([exploded_df, self_map_df], ignore_index=True)

    # Convert all NumPy int64 types to Python native int types for MongoDB compatibility
    final_df = final_df.applymap(lambda x: int(x) if isinstance(x, np.int64) else x)

    # Insert into Mongo Collection
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    
    
    ## Insert into Mongo Collection
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    collection = db[os.environ["MONGODB_MAPPING_COLLECTION"]]
    data_dict = final_df.to_dict("records")
    collection.insert_many(data_dict)
    print(f"[INFO]: Successfully Inserted into Mapping!")
    
def insert_into_document_master(df):
    client = MongoClient(os.environ["MONGODB_IP"], int(os.environ["MONGODB_PORT"]))
    db = client[os.environ["MONGODB_DB"]]
    df = df[['document_name', 'document_word_count', 'document_page_count']].drop_duplicates()
    collection = db[os.environ["MONGODB_DOCUMENTS_MASTER_COLLECTION"]]
    df['_id'] = df['document_name']+".pdf"
    df['insert_timestamp'] = datetime.datetime.now()
    data_dict = df.to_dict("records")
    # Initialize counter for inserted documents
    inserted_count = 0

    # Insert documents only if they don't already exist
    for document in data_dict:
        if collection.count_documents({'_id': document['_id']}, limit=1) == 0:
            collection.insert_one(document)
            inserted_count += 1

    # Logging the result
    print(f"[INFO]: Successfully inserted {inserted_count} new documents into master collection.")
