import os
import faiss
import hashlib
import numpy as np
import json
import time


# Function to calculate a unique hash for text
def calculate_document_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# Normalization function
def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


# Function to load text chunks and their metadata
def load_text_chunks_with_metadata(folder_path):
    """
    Recursively loads text chunks and their metadata from all subfolders.
    Args:
        folder_path (str): Path to the root folder containing subfolders with .txt files.
    Returns:
        list of dict: Each dict contains 'chunk'(text) and 'filename'.
    """
    chunks_with_metadata = []
    for root, _, files in os.walk(folder_path):
        doc_name = os.path.basename(root)
        for filename in sorted(files):
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    chunks_with_metadata.append({
                        "chunk": file.read().strip(),
                        "doc_name": doc_name,
                        "filename": filename
                    })
    return chunks_with_metadata


def get_text_embedding(client, input_text):
    """
    Sends input text to the Mistral embedding model and returns the embedding.
    """
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input_text
    )
    return embeddings_batch_response.data[0].embedding


def create_faiss_index_and_embeddings_if_not_exists(client, faiss_index_file, metadata_file, extracted_text_folder, dimension):
    # Initialize or load FAISS index
    if os.path.exists(faiss_index_file):
        index = faiss.read_index(faiss_index_file)
    else:
        # index = faiss.IndexFlatL2(dimension) # Euclidean distance
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
        # print(f"Processing: {item['filename']}...")
        chunk = item["chunk"]
        doc_name = item["doc_name"]
        filename = item["filename"]
        
        # Calculate document hash
        doc_hash = calculate_document_hash(chunk)

        # Check if already embedded
        if doc_hash not in metadata:
            embedding = get_text_embedding(client, chunk)
            embedding = normalize_vector(np.array(embedding, dtype="float32"))  # Normalize the embedding
            time.sleep(2)
            index.add(np.array([embedding], dtype="float32"))  # Add to FAISS

            # Map FAISS vector ID to metadata
            vector_id = index.ntotal - 1  # Last added vector's ID
            metadata[doc_hash] = {
                "vector_id": vector_id,
                "doc_name": doc_name,
                "filename": filename
            }

            # Save FAISS index and metadata
            print(f"Saving index {faiss_index_file} and metadata {metadata_file}...")
            faiss.write_index(index, faiss_index_file)
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            # print("Already embedded. Skipping...")
            pass

    print("Embedding process completed and metadata saved.")