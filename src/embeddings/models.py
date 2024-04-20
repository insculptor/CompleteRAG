"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/25/2024                              #####
#####                Model used for Summarization                              #####
####################################################################################
"""
## Load Environment Variables
import os
from pathlib import Path
import subprocess
import torch
from angle_emb import AnglE
import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/FinsightRAG/.env'))


def get_gpu_memory_stats():
    """
    Function to Get GPU Memory Statistics
    """
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

def get_device():
    """_summary_

    Returns:
        device: Returns torch.deice as "cuda" is cuda is available, else cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"[INFO]: Using {device=}")
    return device

def get_total_gpu_memory():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes) /(2**30)
    print(f"[INFO]: Total available GPU memory: {gpu_memory_gb} (GB)")




def summarize_mega_chunk(llm_model_name: str, llm_base_path: Path, input_text: str,device):
    """
    Function to Summarize the Mega Chunk created.

    Args:
        llm_model_name (str): The name of the model used for Summarization task.
        llm_base_path (Path): Base path where the model is stored.
        input_text (str): Inout text to be summarised

    Returns:
        summary_text: Generated summary from the input Text.
    """
    llm_dir = os.path.join(llm_base_path, llm_model_name)
    os.makedirs(llm_dir, exist_ok=True)
    #print(f"Loading from {llm_dir=}")

    # Check if model and tokenizer are already downloaded, if not, download
    model_path = Path(llm_dir) / "model"
    try:
        # Attempt to load the summarizer from the local directory
        summarizer = pipeline("summarization", model=str(model_path),device=0)#.to(device)
        #print(f"Loaded model and tokenizer from local directory: {llm_dir}")
    except (EnvironmentError, ValueError) as e:
        print(f"Downloading model and tokenizer because: {e}")
        # Download the model and tokenizer from Hugging Face and save locally
        summarizer = pipeline("summarization", model=llm_model_name).to(device)
        summarizer.model.save_pretrained(model_path)
        summarizer.tokenizer.save_pretrained(model_path)
        print(f"Model and tokenizer downloaded and saved to {model_path}")

    # Determine max_length
    input_length = len(input_text.split())
    max_length = max(100, int(input_length / 2))
    # Generate summary
    summary = summarizer(input_text[:1024], max_length=max_length, min_length=100, do_sample=False)
    summary_text = summary[0]['summary_text']
    #print(f"Summarized Text: {summary_text}")

    # Clear up GPU memory if used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary_text


def get_embedding_model(llm_base_path :Path = Path(os.environ["MODELS_BASE_DIR"]),
                        embedding_model_name :str = os.environ["EMBEDDING_MODEL"]):
    """
        Funtion to get the Embedding model from a local directory.
        
    """     
    embedding_model_dir = os.path.join(llm_base_path,embedding_model_name)
    os.makedirs(embedding_model_dir, exist_ok=True)
    #print(f"Loading from {embedding_model_dir=}")
    # Check if model and tokenizer are already downloaded, if not, download
    model_path = Path(embedding_model_dir) / "model"
    try:
        # Attempt to load the summarizer from the local directory
        angle = AnglE.from_pretrained(model_name_or_path=str(model_path), pooling_strategy='cls').cuda()
        print(f"Loaded model and tokenizer from local directory: {model_path}")
    except (EnvironmentError, ValueError) as e:
        print(f"Downloading model and tokenizer because: {e}")
        # Download the model and tokenizer from Hugging Face and save locally
        angle = pipeline("feature-extraction", embedding_model_name)
        angle.model.save_pretrained(model_path)
        angle.tokenizer.save_pretrained(model_path)
        angle = AnglE.from_pretrained(model_name_or_path=str(model_path), pooling_strategy='cls').cuda()
        print(f"Model and tokenizer downloaded and saved to {model_path}")
    return angle

def check_model_is_available(model_name,model_save_path):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
    # Clear up GPU memory if used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[INFO]: Model downloaded successfully at: {model_save_path=}")
    
def get_reranked_chunks(query, retrieved_chunks,return_documents=True, top_k=5):
    model_name = os.environ["RERANKER"]  
    model_save_path = Path(os.path.join(os.environ["MODELS_BASE_DIR"],model_name))
    model = CrossEncoder(model_save_path,device="cpu")
    reranked_chunks = model.rank(query, retrieved_chunks, return_documents=True, top_k=5)
    return reranked_chunks
