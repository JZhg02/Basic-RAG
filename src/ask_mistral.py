import numpy as np
from embed_documents import get_text_embedding
import time

# def get_text_embedding(input_text):
#     """
#     Sends input text to the Mistral embedding model and returns the embedding.
#     """
#     embeddings_batch_response = client.embeddings.create(
#         model="mistral-embed",
#         inputs=input_text
#     )
#     return embeddings_batch_response.data[0].embedding


def query_with_context(client, question, extracted_text_folder, index, metadata, top_k=5):
    """
    Queries the FAISS index with a question and retrieves relevant chunks with context.
    Args:
        question (str): The query or question.
    Returns:
        list of dict: Retrieved chunks with metadata.
    """
    # Get the embedding of the question
    question_embedding = get_text_embedding(client, question)
    time.sleep(2)

    # Perform similarity search
    _, indices = index.search(np.array([question_embedding], dtype="float32"), top_k)

    # Retrieve relevant chunks and metadata
    relevant_chunks = []
    for vector_id in indices[0]:
        if vector_id != -1:  # Ensure the vector ID is valid
            for doc_hash, meta in metadata.items():
                if meta["vector_id"] == vector_id:
                    with open(f"{extracted_text_folder}{meta["doc_name"]}/{meta["filename"]}", "r", encoding="utf-8") as file:
                        chunk_content = file.read().strip()
                    chunk_info = {
                        "chunk": chunk_content,  # Load content dynamically
                        "doc_name": meta["doc_name"]
                    }
                    relevant_chunks.append(chunk_info)
                    break

    return relevant_chunks


def ask_question(client, prompt, extracted_text_folder, index, metadata, top_k):

    relevant_chunks = query_with_context(client, prompt, extracted_text_folder, index, metadata, top_k)
    # Display and store retrieved context
    contexts = ""
    # print("Relevant Chunks:")
    for chunk in relevant_chunks:
        contexts += f"Document name: {chunk['doc_name']}, Chunk content: \n{chunk['chunk']}\n"
    print(f"List of documents:\n{contexts}")

    # Mistral answer with context 
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages = [
            {
                "role": "system",
                "content": f"You are a cybersecurity expert that will answer question to users. The following are context chunks from the database related to the question:\n{contexts}. Always source information using document names, do not mention those chunks.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content