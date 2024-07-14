import streamlit as st
from service import file_uploader

st.title("Screen 1")

with st.form("Load Data"):
    if load_documents():
        read_from_chroma()      # Instantiate and Read from Chroma  
        ask_for_more_documents  # Ask if there are more documents to ingest
    else:
        mount_google_embedder() # Mount embeddings function for Chroma TASK 1
        ingest_documents()      # Ingest if there are no existing db/ and files TASK 2
        embed_to_chroma()       # Embed and Store into VectorStore, return Vector Store
    
    st.form_submit_button("Submit")
