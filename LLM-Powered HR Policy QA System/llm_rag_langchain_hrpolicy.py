# -*- coding: utf-8 -*-
"""LLM_rag_langchain_hrPolicy.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10zA5ptEVCGKqpUAl0svVRNA8Nty_rV8Y
"""

!pip install -U torch datasets transformers tensorflow langchain playwright html2text sentence_transformers faiss-cpu accelerate peft bitsandbytes trl

!pip install -U langchain-community
!pip install chromadb

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

!huggingface-cli login

#-------Load quantized Mistal 7B
#################################################################
# Tokenizer
#################################################################

#model_name='mistralai/Mistral-7B-Instruct-v0.1'
model_name ='NousResearch/Meta-Llama-3.1-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

#-----Count number of trainable parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

#Build Mistral text generation pipeline
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Loading dataframe
import pandas as pd
df = pd.read_csv('real-text-to-HrPolicy.csv')
print(df.head())

len(df)

df.rename(columns={'text': 'policy'}, inplace=True)

# Checking df length
print('Dataframe Length:', len(df), 'rows')

df = df.dropna() # Dropping empty entries

# Checking df length after dropping empty articles
print('Length After Dropping Empty Values:', len(df), 'rows')

print('Title:', df.policy.iloc[-1])
print('\n\n\n')
print(df.policy.iloc[-1])

# Loading dataframe content into a document
articles = DataFrameLoader(df,
                           page_content_column = "policy")

# Loading entire dataframe into document format
document = articles.load()

# Splitting document into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                chunk_overlap = 20)
splitted_texts = splitter.split_documents(document)

# Loading model to create the embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Creating and indexed database
chroma_database = Chroma.from_documents(splitted_texts,
                                      embedding_model,
                                      persist_directory = 'chroma_db')

# Visualizing the database
chroma_database

# Defining a retriever
retriever = chroma_database.as_retriever()
# Visualizing the retriever
retriever

# Defining a QnA chain
QnA = RetrievalQA.from_chain_type(llm = llm,
                                 chain_type = 'stuff',
                                 retriever = retriever,
                                 verbose = False)

def get_answers(QnA, query):
    answer = QnA.run(query)

    # Extracting Question and Helpful Answer
    question = answer.split('Question: ')[-1].split('\n')[0]
    helpful_answer = answer.split('Helpful Answer: ')[-1].split('\n')[0]

    # Printing the formatted result
    print(f"\033[1mQuery:\033[0m {query}\n")
    print(f"\033[1mQuestion:\033[0m {question}")
    print(f"\033[1mHelpful Answer:\033[0m {helpful_answer}\n")

query = """diwali bonus at this company?"""
get_answers(QnA, query)

query = """different bond periods at this company?"""
get_answers(QnA, query)

