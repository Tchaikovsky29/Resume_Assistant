import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
import streamlit as st

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

#Connecting to the vector db
index_name = "resume-assistant"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

def split_txt_file_by_empty_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    return chunks

file_path = 'about_me.txt' 
chunks = split_txt_file_by_empty_lines(file_path)

retriever.add_texts(chunks)