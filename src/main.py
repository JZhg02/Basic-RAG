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
max_distance_threshold = 0.5728583931922913 # Max distance (or min similarity) threshold for relevant chunks

# ================= User Pormpt ================= 
prompt = "What are the primary objectives of the HIPAA Security Rule in protecting electronic protected health information (ePHI)?"
# prompt = "Does the chunks help answer the following question: 'What are the primary objectives of the HIPAA Security Rule in protecting electronic protected health information (ePHI)?'"
# prompt = "give me some guidelines for Information Technology equipment."
# ===============================================


def main():
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    model = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1") 
    
    # Extract text from each document if not already done
    if not os.path.isdir(extracted_text_folder):
        pdf_to_text.extract_text_from_each_document(documents_folder=documents_folder, extracted_text_folder=extracted_text_folder)

    # Create FAISS index if not already done
    if not os.path.isdir(index_folder):
        # Create FAISS index and add embeddings
        embed_documents.create_faiss_index_and_embeddings_if_not_exists(tokenizer=tokenizer, embeddings_model=model, index_folder=index_folder, extracted_text_folder=extracted_text_folder, dimension=dimension)
        faiss_index_file = os.path.join(index_folder, "faiss_index_with_metadata.bin")
        metadata_file = os.path.join(index_folder, "metadata.json")
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

    client = Mistral(api_key=api_key)

    generator = ask_mistral.ask_question(chat_client=client, tokenizer=tokenizer, embeddings_model=model, prompt=prompt, index=index, metadata=metadata, extracted_text_folder=extracted_text_folder, max_distance_threshold=max_distance_threshold, top_k=top_k)
    full_response = ""
    for chunk in generator:
        full_response += chunk
    print(f"LLM response:\n{full_response}")

if __name__ == "__main__":
    main()
