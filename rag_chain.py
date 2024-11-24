from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from components import main_prompt, rephrase_prompt, ChatGroq, Memory
import streamlit as st
import os

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if "memory" not in st.session_state:
    st.session_state["memory"] = Memory()

@st.cache_resource
def setup_retriever():
    #Connecting to the vector db
    index_name = "resume-assistant"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    bm25_encoder = BM25Encoder().default()
    return index, embeddings, bm25_encoder

index, embeddings, bm25_encoder = setup_retriever()
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
question_rephraser = ChatGroq(GROQ_API_KEY)
llm = ChatGroq(GROQ_API_KEY)

def rag_chain(question):
    rephrase = rephrase_prompt.format(memory = st.session_state["memory"].memory)
    question_rephraser.pass_prompt(rephrase)
    rephrased_question = question_rephraser.invoke(question)

    context = []
    ret = retriever.invoke(rephrased_question)
    for d in ret:
        context.append(d.page_content)

    prompt = main_prompt.format(context = context)
    llm.pass_prompt(prompt)
    response = llm.invoke(rephrased_question)

    st.session_state["memory"].add_mem(human_message=question, AI_message=response)
    return response