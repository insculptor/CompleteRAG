"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/23/2024                              #####
#####           Preprocess the Data For Inserting into a Vector Database.      #####
####################################################################################
"""

## Load Environment Variables
import os
from pathlib import Path
import re
import fitz
import pandas as pd
from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))

def preprocess_text(text):
    """
    Preprocesses the given text by removing extra spaces.
    
    This function performs the following operations on the text:
    - Removes leading and trailing spaces.
    - Replaces multiple consecutive spaces with a single space within the text.
    
    Parameters:
    - text (str): The text to be preprocessed.
    
    Returns:
    - str: The preprocessed text.
    """
    # Strip leading and trailing spaces
    text = text.strip()
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    ## Remove /m
    text = text.replace("\n"," ")
    # Remove the unwanted string from the text
    unwanted_strings = ["Trending Videos", "Close this video player"]
    for unwanted in unwanted_strings:
        text = text.replace(unwanted, "")
    return text



def preprocess_text_math(text):
    """
    Preprocesses the given text to clean up mathematical expressions and descriptions.
    
    This function performs the following operations on the text:
    - Removes leading and trailing spaces.
    - Replaces multiple consecutive spaces with a single space within non-mathematical text.
    - Fixes common misinterpretations of mathematical symbols and expressions.
    
    Parameters:
    - text (str): The text to be preprocessed, containing mathematical expressions.
    
    Returns:
    - str: The preprocessed text with cleaned mathematical expressions.
    """
    # Fix common symbol misinterpretations and alignment issues
    text = text.replace("=", " = ").replace("+", " + ").replace("…", " + … + ")
    text = re.sub(r'\s*=\s*', ' = ', text)
    text = re.sub(r'\s*\+\s*', ' + ', text)
    text = re.sub(r'\s*\\begin{aligned}\s*', '', text)
    text = re.sub(r'\s*\\end{aligned}\s*', '', text)
    text = re.sub(r'\\dotso', '...', text)
    text = re.sub(r'\\textbf', '', text)
    text = re.sub(r'\\text', '', text)
    text = re.sub(r'\\;', '=', text)
    text = re.sub(r'&', '', text)
    text = re.sub(r'GR;_([A-Za-z])', r'GR_\1', text)  # Correcting subscript format

    # Remove leading and trailing spaces and replace multiple spaces with a single space
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        line = ' '.join(line.split())
        processed_lines.append(line)

    return '\n'.join(processed_lines)

def get_data_stats(directory:Path):
    # List to store metadata of all PDFs
    pdf_metadata = []

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                # Full path of the PDF file
                file_path = os.path.join(root, file)
                
                # Try to open and read the PDF file
                try:
                    with fitz.open(file_path) as doc:
                        total_pages = doc.page_count
                        word_count = sum(len(page.get_text("text").split()) for page in doc)
                        
                        # Append metadata to the list
                        pdf_metadata.append({
                            'file_name': file,
                            'total_pages': total_pages,
                            'word_count': word_count
                        })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Convert the list of metadata into a DataFrame
    df = pd.DataFrame(pdf_metadata)
    
    # Return the DataFrame
    return df


# scrape_data_path = Path(os.path.join(os.environ["BASE_SCRAPE_DATA_DIR"],"investopedia_data"))
# print(scrape_data_path)
# df = get_data_stats(scrape_data_path)
# print(df.head())