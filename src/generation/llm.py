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
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))


def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])


def get_model_mem_size(model: torch.nn.Module):
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate model sizes
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)
    model_mem_gb = model_mem_bytes / (1024**3) 

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2), 
            "model_mem_gb": round(model_mem_gb, 2)}
    
def get_llm():
    llm_model_id = os.environ["LLM_MODEL"]
    model_save_path = Path(os.path.join(os.environ["MODELS_BASE_DIR"],llm_model_id))
    use_quantization_config = True
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4but_compute_dtype=torch.float16)
    ## Define Attention
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0) >=8):
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = "sdpa" # Scaled Dot Procust Attention
    print(f"[INFO]: Using Attention: {attn_implementation}")
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_save_path,
                                 torch_dtype=torch.float16,
                                 quantization_config=quantization_config if use_quantization_config else None,
                                 low_cpu_mem_usage=False, # use as much memory as we can
                                 attn_implementation=attn_implementation)#.to(get_device())
    return llm_model,tokenizer

def get_input_tokens(prompt,model):
# Tokenize the input Text (Prompt) - Turn it number nad send to GPU
    tokenizer = AutoTokenizer.from_pretrained(os.environ["LLM_MODEL"])
    input_ids = tokenizer(prompt,
                          return_tensors='pt')#.to("cpu")
    return input_ids

def generate_llm_response(input_ids,llm_model,tokenizer, max_new_tokens=512):
    ## Generate Outputs from Local LLM
    outputs = llm_model.generate(**input_ids,
                            max_new_tokens = 256)
    outputs_decoded = tokenizer.decode(outputs[0])
    return outputs_decoded[outputs_decoded.find("<start_of_turn>model"):].strip("<start_of_turn>model")