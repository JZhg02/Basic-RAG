import numpy as np
from embed_documents import get_text_embedding
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppress error "OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized." from FAISS


def query_with_context(tokenizer, embeddings_model, question, extracted_text_folder, index, metadata, top_k=5):
    """
    Queries the FAISS index with a question and retrieves relevant chunks with context.
    Args:
        question (str): The query or question.
    Returns:
        list of dict: Retrieved chunks with metadata.
    """
    # Get the embedding of the question
    question_embedding = get_text_embedding(question, tokenizer=tokenizer, model=embeddings_model)

    # Perform similarity search
    _, indices = index.search(np.array([question_embedding], dtype="float32"), top_k)

    # Retrieve relevant chunks and metadata
    relevant_chunks = []
    for vector_id in indices[0]:
        if vector_id != -1:  # Ensure the vector ID is valid
            for doc_hash, meta in metadata.items():
                if meta["vector_id"] == vector_id:
                    with open(f"{extracted_text_folder}{meta['doc_name']}/{meta['filename']}", "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        chunk_content = "".join(lines[2:]).strip()
                    chunk_info = {
                        "chunk": chunk_content,  # Load content dynamically
                        "doc_name": meta["doc_name"],
                        "pages": meta["pages"]
                    }
                    relevant_chunks.append(chunk_info)
                    break

    return relevant_chunks


def ask_question(chat_client, tokenizer, embeddings_model, prompt, extracted_text_folder, index, metadata, top_k):

    relevant_chunks = query_with_context(tokenizer, embeddings_model, prompt, extracted_text_folder, index, metadata, top_k)
    # Display and store retrieved context
    contexts = ""
    document_names = []
    # print("Relevant Chunks:")
    for chunk in relevant_chunks:
        contexts += f"Document name: {chunk['doc_name']}\nPages: {chunk['pages']}\nContent:\n{chunk['chunk']}\n"
        document_names.append(chunk['doc_name'])
    print(f"Contexts:\n{contexts}")
    print(f"Document names: {document_names}")

    # Mistral answer with context 
    chat_response = chat_client.chat.complete(
        model="mistral-large-latest",
        messages = [
            {
                "role": "system",
                "content": f"You are a cybersecurity expert that will answer question to users. The following are context chunks from the database related to the question:\n{contexts}. Always source information using document names and pages. Do not mention those chunks as the user is not aware of them.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content