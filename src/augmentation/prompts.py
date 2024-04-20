"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 07/04/2024                              #####
#####                  Desinging and Augmenting Prompts                        #####
####################################################################################
"""
## Load Environment Variables
import os
from pathlib import Path
import pandas as pd
import transformers
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/FinsightRAG/.env'))


def create_prompt(query,reranked_chunks,tokenizer):
    articles_prompt = "\n\n"
    for i,item in enumerate(reranked_chunks):
        articles_prompt += f"Article {i+1}: {item['text']} \nScore:{round(item['score']*100,4)} \n\n"
        
    base_prompt1 = f"""Based on the following relevant articles items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible. 
    
    

    \nRelevant Articles: {articles_prompt}
    User query: {query}"""
    
    base_prompt = f"""Answer the following question in a concise and informative manner:
    
{query}
    
Use the below Articles to answer the above question based on their relevance score. Create a combined answer from the articles:
{articles_prompt}
"""
    
    dialogue_template = [
    {
        "role":"user",
        "content":base_prompt
    }]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt
