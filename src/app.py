import streamlit as st
import pandas as pd
import time 

import pdf_to_text
import embed_documents
import ask_mistral
import os
from mistralai import Mistral
from dotenv import load_dotenv
import faiss
import json
from transformers import AutoTokenizer, AutoModel # initializing a Hugging Face model globally in embed_documents.py, and the multiprocessing module conflicts with this
from transformers import logging

# Set the logging level to ERROR to suppress warnings
logging.set_verbosity_error() 
# Warning: 
# Some weights of RobertaModel were not initialized from the model checkpoint at SynamicTechnologies/CYBERT and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

load_dotenv()

# Config
documents_folder = "data/documents/"
extracted_text_folder = "data/extracted_text/"
# os.makedirs(extracted_text_folder, exist_ok=True) # Create the output folder if it doesn't exist

index_folder = "data/faiss_vector_base/"
# os.makedirs(index_folder, exist_ok=True) 

api_key = os.environ["MISTRAL_API_KEY"]
dimension = 768 # Embedding vector size for most models based on BERT 
top_k = 10  # Number of relevant chunks to retrieve


# Function to process the uploaded Excel file
def process_excel(file):
    """
    Simulates processing the uploaded Excel file.
    Replace this with your actual processing logic.
    """
    time.sleep(3)  # Simulate a delay in processing
    df = pd.read_excel(file)  # Read the uploaded file
    # Example processing: Adding a new column
    df["Processed"] = df.iloc[:, 0].apply(lambda x: f"Processed: {x}")
    return df

# RAG initialization
def initialize_rag():
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    model = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1") 

    # Extract text from each document if not already done
    if not os.path.isdir(extracted_text_folder):
        pdf_to_text.extract_text_from_each_document(documents_folder=documents_folder, extracted_text_folder=extracted_text_folder)

    # Create FAISS index if not already done
    if not os.path.isdir(index_folder):
        # Create FAISS index and add embeddings
        embed_documents.create_faiss_index_and_embeddings_if_not_exists(tokenizer=tokenizer, embeddings_model=model, index_folder=index_folder, faiss_index_file=faiss_index_file, metadata_file=metadata_file, extracted_text_folder=extracted_text_folder, dimension=dimension)
    else: 
        faiss_index_file = os.path.join(index_folder, "faiss_index_with_metadata.bin")
        metadata_file = os.path.join(index_folder, "metadata.json")
    
    # Load FAISS index
    if not os.path.exists(faiss_index_file):
        raise FileNotFoundError("FAISS index file not found.")
    index = faiss.read_index(faiss_index_file)
    # Load metadata
    if not os.path.exists(metadata_file):
        raise FileNotFoundError("Metadata file not found.")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return tokenizer, model, index, metadata


# Page 1: Single Question Page
def single_question_page(client, tokenizer, model, index, metadata, extracted_text_folder, top_k):
    st.subheader("Single Question")

    # Accept user input
    question = st.text_input("Ask me anything...")

    # Display assistant response
    if question:
        with st.spinner("Thinking..."):
            response = ask_mistral.ask_question(
                chat_client=client, 
                tokenizer=tokenizer, 
                embeddings_model=model, 
                prompt=question, 
                index=index, 
                metadata=metadata, 
                extracted_text_folder=extracted_text_folder, 
                top_k=top_k
            )
            st.write(response)


# Page 2: Chat Page
def chat_page(client, tokenizer, model, index, metadata, extracted_text_folder, top_k):
    st.subheader("Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [] # [{"role": "assistant", "content": "Comment puis-je vous aider?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            full_response = ""
            placeholder = st.empty()
            stream = ask_mistral.ask_question(
                chat_client=client, 
                tokenizer=tokenizer, 
                embeddings_model=model, 
                prompt=prompt, 
                index=index, 
                metadata=metadata, 
                extracted_text_folder=extracted_text_folder, 
                top_k=top_k,
                chat_history=st.session_state.messages
            )
            for chunk in stream:
                full_response += chunk  # Accumulate the full response
                placeholder.markdown(full_response)  # Display all content in one block

        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Page 3: Dataset page
def dataset_page(client, tokenizer, model, index, metadata, extracted_text_folder, top_k):
    st.subheader("Dataset")
    
    # Upload file section
    uploaded_file = st.file_uploader("Upload data (Excel format)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Display a spinner while processing
        with st.spinner("Processing the file..."):
            data = pd.read_excel(uploaded_file, header=0)
            data["LLM_Answers"] = data["Questions"].apply(
                lambda x: "".join(
                    chunk for chunk in ask_mistral.ask_question(
                        chat_client=client, 
                        tokenizer=tokenizer, 
                        embeddings_model=model, 
                        prompt=x, 
                        index=index, 
                        metadata=metadata, 
                        extracted_text_folder=extracted_text_folder, 
                        top_k=top_k
                    )
                )
            )

        # Show success message and display the resulting dataframe
        st.success("File processed successfully!")
        st.dataframe(data["Questions", "Reponses"])  # Display the dataframe in Streamlit
    

# Main function with tabs
def main():
    # ----------------- RAG Initialization -----------------
    if "rag_initialized" not in st.session_state:
        with st.spinner("Initializing, please wait..."):
            try:
                tokenizer, model, index, metadata = initialize_rag()
                # Store the initialized objects in session_state
                st.session_state.rag_initialized = True
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.index = index
                st.session_state.metadata = metadata
                st.toast("Initialization successful!", icon="✅")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
    # Retrieve the initialized objects from session_state
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    index = st.session_state.index
    metadata = st.session_state.metadata

    # Initialize Mistral client
    client = Mistral(api_key=api_key)

    # ----------------- Front end app -----------------
    st.title("Cyber IA")

    # Sidebar navigation
    with st.sidebar:
        page = st.radio(
            "Navigation",
            options=["Single Question", "Chat", "Dataset"]
        )


    # Render content based on the selected page
    if page == "Single Question":
        single_question_page(client, tokenizer, model, index, metadata, extracted_text_folder, top_k)
    elif page == "Chat":
        chat_page(client, tokenizer, model, index, metadata, extracted_text_folder, top_k)
    elif page == "Dataset":
        dataset_page(client, tokenizer, model, index, metadata, extracted_text_folder, top_k)


if __name__ == "__main__":
    main()
