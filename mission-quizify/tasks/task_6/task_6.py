import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator

if __name__ == "__main__":
    st.header("Quizzify")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "sample-gemini-424615",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quizzify")
        document_processor = DocumentProcessor()
        document_processor.ingest_documents()
        embed_client = EmbeddingClient(**embed_config)
        chroma_creator = ChromaCollectionCreator(document_processor, embed_client)

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Enter the topic for the quiz")
            num_questions = st.slider("Select the number of questions", min_value=1, max_value=10)
            
            document = None
            
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:
                document = chroma_creator.create_chroma_collection()

                document = chroma_creator.query_chroma_collection(topic_input)
                
    if document:
        screen.empty()
        with st.container():
            st.header("Query Chroma for Topic, top Document: ")
            st.write(document)