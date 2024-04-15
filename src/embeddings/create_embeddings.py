"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/24/2024                              #####
#####                Create Mega Chunks and Chunks from Data                   #####
####################################################################################
"""

## Load Environment Variables
import os
from pathlib import Path
import unicodedata
from time import perf_counter as timer
from tqdm.auto import tqdm
import torch
import fitz
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.embeddings.models import summarize_mega_chunk,get_embedding_model
from src.embeddings.vectorstore import add_to_faiss_l2_and_hnsw_indices
from src.embeddings.database import insert_into_mongodb,update_chunk_mapping_collection,insert_into_document_master
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))


def get_document_pagewise_data(file_path:Path):
    doc = fitz.open(file_path)
    pagewise_data = []
    text=" "
    document_text = " "
    for num,page in tqdm(enumerate(doc)):
        text = page.get_text().replace("\n"," ").strip()
        normalized_text = unicodedata.normalize('NFKD', text)
        text = normalized_text.encode('ascii', 'ignore').decode('ascii')
        document_text += text + " "
        pagewise_data.append({
            "page_number": num + 1,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count": len(text.split(". ")),
            "page_token_count": len(text) // 4,
            "page_text": text,
            "document_name": os.path.basename(file_path).replace(".pdf","")
            })
    doc.close()
    ## Add Document Level Metadata
    for doc in pagewise_data:
        doc["document_word_count"] = len(document_text)
        doc["document_page_count"] = num + 1
    return pagewise_data


def get_all_documents_data(documents_dir_path : Path):
    """
    This function gets calls get_document_pagewise_data functions for all the docuements.
    
    Args:
        documents_dir_path (Path): _description_

    Returns:
        _type_: _description_
    """
    all_documents_data = []
    docs_list = os.listdir(documents_dir_path)#[:10]
    print(f"[INFO]: Total Documents to be Processed: {len(docs_list)}")
    for item in tqdm(docs_list):
        file_path = Path(os.path.join(documents_dir_path,item))
        all_documents_data.append(get_document_pagewise_data(file_path))
    
    # Flatten the list
    flattened_list = []
    for sublist in all_documents_data:
        # Loop through each dictionary in the inner lists and add it to the flattened_list
        for d in sublist:
            flattened_list.append(d)
    
    return pd.DataFrame(flattened_list)

def find_overlap(str1: str, str2: str) -> int:
    """
    Finds the length of the largest overlap between the end of str1 and the start of str2.
    
    Parameters:
    - str1 (str): The first string.
    - str2 (str): The second string.
    
    Returns:
    - int: The length of the overlap.
    """
    min_length = min(len(str1), len(str2))
    for i in range(1, min_length + 1):
        if str1[-i:] == str2[:i]:
            return i
    return 0

def combine_strings_remove_overlap(str_list: list) -> str:
    """
    Combines a list of strings by removing overlapping text.
    
    Parameters:
    - str_list (list): A list of strings with possible overlaps.
    
    Returns:
    - str: A single combined string with overlaps removed.
    """
    if not str_list:
        return ""

    # Initialize the combined string with the first string in the list
    combined_string = str_list[0]

    # Iterate through the list, combining strings with overlap removed
    for i in range(1, len(str_list)):
        overlap_length = find_overlap(combined_string, str_list[i])
        combined_string += str_list[i][overlap_length:]

    return combined_string


def create_chunks(df,
                  chunk_size: int = int(os.environ.get("CHUNK_SIZE", 256)),
                  chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP_SIZE", 50)),
                  mega_chunk_multiplier: int = int(os.environ.get("MEGA_CHUNK_MULTIPLIER", 4))):
    print(f"[INFO]: Creating Chunks Using: chunk_size={chunk_size} | chunk_overlap={chunk_overlap} | mega_chunk_multiplier={mega_chunk_multiplier}")

    # Assuming RecursiveCharacterTextSplitter is implemented correctly
    chunk_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Split text into chunks
    df['chunks'] = df['page_text'].apply(lambda x: chunk_text_splitter.split_text(x))

    # Create a list of mega chunks, each with its constituent chunks
    def create_mega_chunks_with_chunks(chunks, multiplier):
        mega_chunks_with_chunks = []
        for i in range(0, len(chunks), multiplier):
            mega_chunk_with_chunks = chunks[i:i + multiplier]
            mega_chunks_with_chunks.append({'mega_chunk': ''.join(mega_chunk_with_chunks), 'chunks': mega_chunk_with_chunks})
        return mega_chunks_with_chunks

    df['mega_chunks_with_chunks'] = df['chunks'].apply(lambda x: create_mega_chunks_with_chunks(x, mega_chunk_multiplier))
    
    # Explode DataFrame by 'mega_chunks_with_chunks'
    df = df.explode('mega_chunks_with_chunks').reset_index(drop=True)
    
    # Extract 'mega_chunk' and 'chunks' from the exploded dictionaries
    df['mega_chunk'] = df['mega_chunks_with_chunks'].apply(lambda x: x['mega_chunk'])
    df['chunks_in_mega'] = df['mega_chunks_with_chunks'].apply(lambda x: x['chunks'])
    
    # Assign mega_chunk_number for enumeration within each document/page
    df['mega_chunk_number'] = df.groupby(['document_name', 'page_number']).cumcount() + 1

    # Calculate the number of mega chunks per page
    df['page_mega_chunk_count'] = df.groupby(['document_name', 'page_number'])['mega_chunk_number'].transform('max')

    # Specify columns to include in the final DataFrame
    df_columns = ['document_name', 'document_word_count', 'document_page_count', 'page_number', 'page_char_count', 'page_word_count',
                  'page_sentence_count', 'page_token_count', 'page_text', 'page_mega_chunk_count', 'mega_chunk_number', 'mega_chunk', 'chunks_in_mega']
    df = df[df_columns]
    df_final = df.rename(columns={'chunks_in_mega': 'chunks'})

    return df_final


def add_summary_to_dataframe(df, llm_model_name,llm_base_path,device):
    """
    Takes a DataFrame with a 'mega_chunk' column and applies a summarizer function to each entry,
    adding the summaries to a new column named 'mega_chunk_summary'.

    Parameters:
    - df: The DataFrame containing the 'mega_chunk' column.
    - summarizer_function: A function that takes text as input and returns its summary.
    """
    # Check if 'mega_chunk' column exists
    if 'mega_chunk' not in df.columns:
        raise ValueError("'mega_chunk' column not found in the DataFrame")

    # Initialize an empty list to store summaries
    summaries = []
    print(f"[INFO]: Starting Summary Generation...")
    print(f"[INFO]: Total Records to be processed: {df.shape}")
    # Iterate over each row in the DataFrame
    for _, row in tqdm(df.iterrows()):
        # Extract the 'mega_chunk' content
        print(f"[INFO]: Generating Summary for {row['document_name']} Page: {row['page_number']}")
        content = row['mega_chunk']
        
        # Ensure mega_chunk is a string or a list of strings
        if isinstance(content, list):
            # If it's a list, join the elements into a single string
            content = ' '.join(content)
        elif not isinstance(content, str):
            # If it's neither a list nor a string, raise an error
            raise ValueError("Each 'mega_chunk' entry must be of type str or type list")

        # Apply the summarizer function to the content
        summary = summarize_mega_chunk(llm_model_name, llm_base_path, content,device)
        # Append the summary to the list of summaries
        summaries.append(summary)

    # Add the list of summaries as a new column to the DataFrame
    df['mega_chunk_summary'] = summaries
    

def generate_embeddings(df, angle):
    """
    Generates embeddings for the 'mega_chunk_summary' and each element in 'chunks'
    in the given DataFrame using the provided 'angle' model's encode() function.
    
    Parameters:
    - df: A pandas DataFrame with columns 'mega_chunk_summary' (str) and 'chunks_list' (list of str).
    - angle: An embedding model instance with an encode() method for generating embeddings.
    
    Returns:
    - A new DataFrame with additional columns for embeddings: 'mega_chunk_summary_embedding' and 'chunks_embedding_list'.
    """
    
    # Initialize empty lists to store embeddings
    mega_chunk_summary_embeddings = []
    chunks_embedding_list = []
    
    # Iterate over rows in the DataFrame
    for _, row in tqdm(df.iterrows()):
        # Generate embedding for the mega_chunk_summary

        mega_chunk_summary_embedding = angle.encode(str(row['mega_chunk_summary']), to_numpy=True)
        mega_chunk_summary_embeddings.append(mega_chunk_summary_embedding[0])
        print(len(mega_chunk_summary_embedding[0]))
        
        # Initialize a list to store embeddings for the current row's chunks_list
        current_chunks_embeddings = []
        
        # Iterate over each chunk in the chunks_list and generate its embedding
        for chunk in row['chunks']:
            chunk_embedding = angle.encode(chunk, to_numpy=True)
            print("**************")
            print(len(chunk_embedding[0]))
            current_chunks_embeddings.append(chunk_embedding[0])
        
        # Append the list of embeddings for the current row's chunks_list to the main list
        chunks_embedding_list.append(current_chunks_embeddings)
    
    # Add the new columns with embeddings to the DataFrame
    df['mega_chunk_summary_embedding'] = mega_chunk_summary_embeddings
    df['chunks_embedding_list'] = chunks_embedding_list
    return df


if __name__=="__main__":
    start_time = timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed_data_path = Path(os.environ["PREPROCESSED_DATA_DIR"])
    processed_data_path = Path(os.environ["PROCESSED_DATA_DIR"])
    print(f"[INFO]: {preprocessed_data_path=} | {processed_data_path=}")
    all_documents_data = get_all_documents_data(preprocessed_data_path)
    all_documents_data = create_chunks(all_documents_data)
    llm_base_path = Path(os.environ["MODELS_BASE_DIR"])
    llm_model_name = os.environ["SUMMARIZATION_MODEL"]
    print(f" {llm_base_path=} | {llm_model_name=}")
    add_summary_to_dataframe(all_documents_data, llm_model_name,llm_base_path,device)
    angle = get_embedding_model().to(device)
    all_documents_data = generate_embeddings(all_documents_data, angle)
    vectorstore_base_path = os.environ["VECTORSTORE_BASE_DIR"]
    print(f"{vectorstore_base_path=}")
    all_documents_data = add_to_faiss_l2_and_hnsw_indices(all_documents_data, vectorstore_base_path)
    cols = ['document_name', 'document_word_count', 'document_page_count','page_number', 'page_char_count', 'page_word_count','page_sentence_count', 'page_token_count', 'page_text','page_mega_chunk_count', 'mega_chunk_number', 'mega_chunk','mega_chunk_summary', 'mega_chunk_summary_embedding_index','chunks','chunks_embedding_list_index']
    all_documents_data = all_documents_data[cols]
    insert_into_mongodb(all_documents_data)
    update_chunk_mapping_collection(all_documents_data)
    insert_into_document_master(all_documents_data)
    all_documents_data.to_json(os.path.join(processed_data_path,"all_chunks2.json"), orient="records", lines=True)
    all_documents_data.to_json(os.path.join(processed_data_path,"all_chunks_readable2.json"), orient="records", lines=True,indent=4)
    end_time = timer()
    print(f"[INFO]: Time taken to create Embeddings: {end_time - start_time:.5f} seconds.")
