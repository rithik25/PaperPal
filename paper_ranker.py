import re
import numpy as np
import networkx as nx

from rank_bm25 import BM25Okapi
from openai import OpenAI
# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     HfArgumentParser,
#     TrainingArguments,
#     pipeline,
#     logging,
# )

import requests




def tokenize(input_string):
    tokenized_string = input_string.lower()
    tokenized_string = re.sub(r'\[.*?\]', '', tokenized_string)
    tokenized_string = re.sub(r'[^a-zA-Z]', ' ', tokenized_string)
    tokens = tokenized_string.split()
    filtered_tokens = [word for word in tokens if re.search(r'[a-zA-Z]', word)]

    return filtered_tokens

def calc_scores(query, hits_scores, bm25_model):

    tokenized_query = tokenize(query)
    doc_scores = np.multiply(bm25_model.get_scores(tokenized_query), np.array(list(hits_scores[0].values())) + 0.0005)
    doc_ranking = np.argsort(-doc_scores)
    return doc_scores, doc_ranking

def query_chatgpt(query, hits_scores, bm25_model, doc_texts, doc_authors, doc_titles):
    print(query)
    scores, doc_ranking = calc_scores(query, hits_scores, bm25_model)
    prompt_context = ""
    for i, x in enumerate(doc_ranking[:5]):
        # print(f"{x}: {scores[x]}, {doc_texts[x]}")
        prompt_context += f"Paper Title: {doc_titles[x]}, Authored by: {doc_authors[x]}, Content: {doc_texts[x]}\n"
        

    client = OpenAI(api_key="YOUR OPENAI API KEY")
    print((prompt_context))

    response = client.chat.completions.create(
    model = "gpt-3.5-turbo-0125",
    temperature = 0.8,
    max_tokens = 3000,
    #response_format={ "type": "json_object" },
    messages = [
        {"role": "system", "content": "You are a helpful assistent that can write literature reviews for research papers."},
        {"role": "user", "content": f"Write a literature review about {query} using the following papers: {prompt_context}. \n Use the format Paper Title, (Paper author) :\n literarture review"}
    ])
    print(response)

    return response.choices[0].message.content

    # url = "https://api.openai.com/v1/chat/completions"
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json"
    # }
    # payload = {
    #     "model": "gpt-3.5-turbo",  # or "gpt-4" depending on your access
    #     "messages": [
    #         {"role": "system", "content": "You are a helpful assistent that can write literature reviews for research papers."},
    #         {"role": "user", "content": f"Write a literature review about {query} using the following papers: {prompt_context}"}
    #     ]
    # }
    # response = requests.post(url, json=payload, headers=headers)

    # return response.json()



# Replace 'your_api_key_here' with your actual OpenAI API key

#prompt = "Hello, who are you?"

# response = query_chatgpt('introduce yourselves')
# print(response)

if __name__ == "__main__":
    query = input("Enter your question: ")

    prompt_context = ''
    scores, doc_ranking = calc_scores(query)
    for i, x in enumerate(doc_ranking[:5]):
        print(f"{x}: {scores[x]}, {doc_texts[x]}")
        prompt_context += f"paper {i+1}: {doc_texts[x]}\n"


    
    # # Model from Hugging Face hub
    # base_model = "meta-llama/Llama-2-13b-chat-hf"

    # # Cache directory for model and dataset
    # cache_dir = "/scratch/user/kangda"

    # ################################################################################
    # # bitsandbytes parameters
    # ################################################################################

    # # Activate 4-bit precision base model loading
    # use_4bit = True

    # # Compute dtype for 4-bit base models
    # bnb_4bit_compute_dtype = "float16"

    # # Quantization type (fp4 or nf4)
    # bnb_4bit_quant_type = "nf4"

    # # Activate nested quantization for 4-bit base models (double quantization)
    # use_nested_quant = False

    # # Load the entire model on the GPU 0
    # device_map = 'auto'

    # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant,
    # )

    # # Check GPU compatibility with bfloat16
    # if compute_dtype == torch.float16 and use_4bit:
    #     major, _ = torch.cuda.get_device_capability()
    #     if major >= 8:
    #         print("=" * 80)
    #         print("Your GPU supports bfloat16: accelerate training with bf16=True")
    #         print("=" * 80)

    # # Load base model
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     quantization_config=bnb_config,
    #     device_map=device_map,
    #     cache_dir=cache_dir
    # )
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1

    # # Load LLaMA tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache_dir)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # model.eval()


    # messages = [{"role": "system", "content": "You are a helpful assistent that can write literature reviews for research papers."},
    #             {"role": "user", "content": f"Write a literature review about {query} using the following papers: {prompt_context}"}]

    # model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    # with torch.no_grad():
    #     pred = tokenizer.decode(model.generate(model_input)[0])
    #     print(pred)
    #     print('\n\n')
    #     pred = pred.split('[/INST] ')[-1]
    #     print(pred)

