import pdf_to_text
import embed_documents
import ask_mistral
import os
from mistralai import Mistral
from dotenv import load_dotenv
import faiss
import json

load_dotenv()

# Config
documents_folder = "data/documents/"
extracted_text_folder = "data/extracted_text/"
os.makedirs(extracted_text_folder, exist_ok=True) # Create the output folder if it doesn't exist

index_folder = "data/faiss_vector_base/"
os.makedirs(index_folder, exist_ok=True) 

api_key = os.environ["MISTRAL_API_KEY"]
dimension = 1024 # Embedding vector size (mistral-embed model only accepts 1024)
top_k = 10  # Number of relevant chunks to retrieve


# ================= User Pormpt ================= 
prompt = "What are the primary objectives of the HIPAA Security Rule in protecting electronic protected health information (ePHI)?"
# ===============================================


def main():
    pdf_to_text.extract_text_from_each_document(documents_folder=documents_folder, extracted_text_folder=extracted_text_folder)
    
    faiss_index_file = os.path.join(index_folder, "faiss_index_with_metadata.bin")
    metadata_file = os.path.join(index_folder, "metadata.json")

    client = Mistral(api_key=api_key)
    embed_documents.create_faiss_index_and_embeddings_if_not_exists(client=client, faiss_index_file=faiss_index_file, metadata_file=metadata_file, extracted_text_folder=extracted_text_folder, dimension=dimension)

    # Load FAISS index
    if not os.path.exists(faiss_index_file):
        raise FileNotFoundError("FAISS index file not found.")
    index = faiss.read_index(faiss_index_file)
    # Load metadata
    if not os.path.exists(metadata_file):
        raise FileNotFoundError("Metadata file not found.")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    answer = ask_mistral.ask_question(client=client, prompt=prompt, index=index, metadata=metadata, extracted_text_folder=extracted_text_folder, top_k=top_k)
    print(f"LLM answer:\n{answer}")

if __name__ == "__main__":
    main()
    