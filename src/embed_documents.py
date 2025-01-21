import os
import faiss
import hashlib
import numpy as np
import json
import torch
import torch.nn.functional as F


# Function to calculate a unique hash for text
def calculate_document_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Function to load text chunks and their metadata
def load_text_chunks_with_metadata(folder_path):
    chunks_with_metadata = []
    for root, _, files in os.walk(folder_path):
        doc_name = os.path.basename(root)
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    pages = lines[1].strip() 
                    content = "".join(lines[2:]).strip()
                    chunks_with_metadata.append({
                        "chunk": content,
                        "doc_name": doc_name,
                        "filename": filename,
                        "pages": pages
                    })
    return chunks_with_metadata


# Function to get text embeddings using Hugging Face model
def get_text_embedding(text, tokenizer, model):
    """
    Generates text embeddings using a Hugging Face model.
    """
    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        # max_length=512,  # Model's max token limit (514 includes special tokens)
        truncation=True,  # Truncate sequences longer than max_length
    )
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
        # Use mean pooling for the embedding
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy().squeeze()


# Function to create FAISS index and add embeddings
def create_faiss_index_and_embeddings_if_not_exists(tokenizer, embeddings_model, index_folder, faiss_index_file, metadata_file, extracted_text_folder, dimension):
    # Create FAISS index folder if it doesn't exist
    os.makedirs(index_folder, exist_ok=True) 

    # Initialize or load FAISS index
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
    else:
        index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)

    # Load metadata if it exists
    if os.path.exists(metadata_file):
        print("Loading metadata...")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
        print("No metadata found. Starting from scratch.")

    # Load text chunks with metadata
    chunks_with_metadata = load_text_chunks_with_metadata(extracted_text_folder)

    # Process chunks and add embeddings
    for item in chunks_with_metadata:
        chunk = item["chunk"]
        doc_name = item["doc_name"]
        filename = item["filename"]
        pages = item["pages"]
        
        # Calculate document hash
        doc_hash = calculate_document_hash(chunk)

        # Check if already embedded
        if doc_hash not in metadata:
            embedding = get_text_embedding(chunk, tokenizer=tokenizer, model=embeddings_model)
            embedding = np.array([embedding], dtype="float32")

            index.add(embedding)  # Add to FAISS

            # Map FAISS vector ID to metadata
            vector_id = index.ntotal - 1  # Last added vector's ID
            metadata[doc_hash] = {
                "vector_id": vector_id,
                "doc_name": doc_name,
                "filename": filename,
                "pages": pages
            }

            # Save FAISS index and metadata
            print(f"Saving index and metadata of file {filename}...")
            faiss.write_index(index, faiss_index_file)
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            print(f"{filename} already embedded. Skipping...")
            pass

    print("Embedding process completed and metadata saved.")
