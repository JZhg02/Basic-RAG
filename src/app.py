import streamlit as st
import pandas as pd
import time 
import plotly.express as px

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
documents_folder = "data-deploy/documents/"
extracted_text_folder = "data-deploy/extracted_text/"
# os.makedirs(extracted_text_folder, exist_ok=True) # Create the output folder if it doesn't exist

index_folder = "data-deploy/faiss_vector_base/"
# os.makedirs(index_folder, exist_ok=True) 

api_key = os.environ["MISTRAL_API_KEY"]
dimension = 768 # Embedding vector size for most models based on BERT 
top_k = 10  # Number of relevant chunks to retrieve
max_distance_threshold = 0.5728583931922913 # Max distance (or min similarity) threshold for relevant chunks


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

    # # Create FAISS index if not already done
    # if not os.path.isdir(index_folder):
    #     # Create FAISS index and add embeddings
    embed_documents.create_faiss_index_and_embeddings_if_not_exists(tokenizer=tokenizer, embeddings_model=model, index_folder=index_folder, extracted_text_folder=extracted_text_folder, dimension=dimension)
    faiss_index_file = os.path.join(index_folder, "faiss_index_with_metadata.bin")
    metadata_file = os.path.join(index_folder, "metadata.json")
    # else: 
    #     faiss_index_file = os.path.join(index_folder, "faiss_index_with_metadata.bin")
    #     metadata_file = os.path.join(index_folder, "metadata.json")
    
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
def single_question_page(client, tokenizer, model, index, metadata, extracted_text_folder, max_distance_threshold, top_k):
    st.subheader("Single Question")

    # Accept user input
    question = st.text_input("Ask me anything.")

    # Display assistant response
    if question:
        with st.spinner("Thinking..."):
            response, documents = ask_mistral.ask_question_and_get_sources(
                chat_client=client, 
                tokenizer=tokenizer, 
                embeddings_model=model, 
                prompt=question, 
                index=index, 
                metadata=metadata, 
                extracted_text_folder=extracted_text_folder, 
                max_distance_threshold = max_distance_threshold,
                top_k=top_k
            )
        st.write(response)
        # Expanders for all documents
        st.write("**Retrieved documents:**")
        for doc in documents:
            with st.expander(f"Document: {doc['doc_name']}"):
                st.write(f"Pages: {doc['pages']}")
                st.write(doc["chunk"])


# Page 2: Chat Page
def chat_page(client, tokenizer, model, index, metadata, extracted_text_folder, max_distance_threshold, top_k):
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
                max_distance_threshold=max_distance_threshold,
                top_k=top_k,
                chat_history=st.session_state.messages
            )
            for chunk in stream:
                full_response += chunk  # Accumulate the full response
                placeholder.markdown(full_response)  # Display all content in one block

        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Page 3: Dataset page
def dataset_page(client, tokenizer, model, index, metadata, extracted_text_folder, max_distance_threshold, top_k):
    st.subheader("Dataset")

    # Display dataset instructions
    # Must contain "Questions" and "Answers" columns
    # "LLM_Answers" will be generated and verified against the "Answers" column
    st.write("The uploaded file must contain 'Questions' and 'Answers' columns for generating LLM answers.")
    
    # User mode selection
    mode = st.selectbox("Choose an action:", ["Generate LLM Answers", "Correct LLM Answers", "Get accuracy"])

    # Upload file section
    uploaded_file = st.file_uploader("Upload file data (Excel format)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        
        # Read the uploaded file
        data = pd.read_excel(uploaded_file, header=0)

        if "generated_answers" not in st.session_state:
            st.session_state["generated_answers"] = None
        if "corrected_answers" not in st.session_state:
            st.session_state["corrected_answers"] = None

        if mode == "Generate LLM Answers":
            if st.session_state["generated_answers"] is None:
                with st.spinner("Generating LLM answers..."):
                    data["LLM_Answers"] = data["Questions"].apply(
                        lambda x: "".join(
                            chunk for chunk in ask_mistral.ask_question(
                                chat_client=client, tokenizer=tokenizer, embeddings_model=model, prompt=x, index=index, metadata=metadata, extracted_text_folder=extracted_text_folder, max_distance_threshold=max_distance_threshold, top_k=top_k
                            )
                        )
                    )
                    # Store the generated answers in session_state
                    st.session_state["generated_answers"] = data.copy()
                st.success("LLM answers generated successfully!")
            else:
                data = st.session_state["generated_answers"]
        
            st.dataframe(data)
            
            # Option to verify answers after generation
            if st.button("Correct LLM Answers"):
                if st.session_state["corrected_answers"] is None:
                    with st.spinner("Correcting LLM answers..."):
                        data[["Verification", "LLM_correction"]] = data.apply(
                            lambda row: pd.Series(ask_mistral.is_llm_answer_correct(
                                chat_client=client, 
                                question=row["Questions"], 
                                answer=row["Answers"], 
                                llm_answer=row["LLM_Answers"]
                            )), 
                            axis=1
                        )
                        # Update session state with verified answers
                        st.session_state["corrected_answers"] = data.copy()
                    st.success("LLM answers verified successfully!")
                else:
                    data = st.session_state["corrected_answers"]

                st.dataframe(data)

                with st.spinner("Computing hallucination rate..."):
                    # Calculate and display accuracy chart
                    accuracy = data["Verification"].value_counts()
                    accuracy_chart = px.pie(
                        names=["Correct", "Incorrect"],
                        values=[accuracy.get("Correct", 0), accuracy.get("Incorrect", 0)],
                        title="LLM Answer Accuracy"
                    )
                    st.plotly_chart(accuracy_chart)
                    # st.write(f"Accuracy: {accuracy.get("Correct", 0) * 100 / len(data)}%")
                    st.write(f"Hallucination rate: {accuracy.get('Incorrect', 0) * 100 / len(data):.2f}%")

        elif mode == "Correct LLM Answers":
            with st.spinner("Correcting LLM answers..."):
                data[["Verification", "LLM_correction"]] = data.apply(
                    lambda row: pd.Series(ask_mistral.is_llm_answer_correct(
                        chat_client=client, 
                        question=row["Questions"], 
                        answer=row["Answers"], 
                        llm_answer=row["LLM_Answers"]
                    )), 
                    axis=1
                )
                # Update session state with verified answers
                st.session_state["corrected_answers"] = data.copy()
            st.success("LLM answers verified successfully!")

            st.dataframe(data)
            with st.spinner("Computing hallucination rate..."):
                # Calculate and display accuracy chart
                accuracy = data["Verification"].value_counts()
                accuracy_chart = px.pie(
                    names=["Correct", "Incorrect"],
                    values=[accuracy.get("Correct", 0), accuracy.get("Incorrect", 0)],
                    title="LLM Answer Accuracy"
                )
                st.plotly_chart(accuracy_chart)
                # st.write(f"Accuracy: {accuracy.get("Correct", 0) * 100 / len(data)}%")
                st.write(f"Hallucination rate: {accuracy.get("Incorrect", 0) * 100 / len(data)}%")

        elif mode == "Get accuracy":
            with st.spinner("Computing hallucination rate..."):
                # Calculate and display accuracy chart
                accuracy = data["Verification"].value_counts()
                accuracy_chart = px.pie(
                    names=["Correct", "Incorrect"],
                    values=[accuracy.get("Correct", 0), accuracy.get("Incorrect", 0)],
                    title="LLM Answer Accuracy"
                )
                st.plotly_chart(accuracy_chart)
                # st.write(f"Accuracy: {accuracy.get("Correct", 0) * 100 / len(data)}%")
                st.write(f"Hallucination rate: {accuracy.get("Incorrect", 0) * 100 / len(data)}%")


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
            "Pages",
            options=["Dataset", "Single Question", "Chat"]
        )

    # Render content based on the selected page
    if page == "Single Question":
        single_question_page(client, tokenizer, model, index, metadata, extracted_text_folder, max_distance_threshold, top_k)
    elif page == "Chat":
        chat_page(client, tokenizer, model, index, metadata, extracted_text_folder, max_distance_threshold, top_k)
    elif page == "Dataset":
        dataset_page(client, tokenizer, model, index, metadata, extracted_text_folder, max_distance_threshold, top_k)


if __name__ == "__main__":
    main()
