import numpy as np
from embed_documents import get_text_embedding
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppress error "OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized." from FAISS


def query_with_context(tokenizer, embeddings_model, question, extracted_text_folder, index, metadata, max_distance_threshold, top_k=5): 
    """
    Queries the FAISS index with a question and retrieves relevant chunks with context.
    
    Args:
        tokenizer: Tokenizer for embedding model.
        embeddings_model: The embedding model.
        question (str): The query or question.
        extracted_text_folder (str): Folder containing extracted text files.
        index: FAISS index.
        metadata (dict): Metadata containing vector mappings.
        max_distance_threshold (float): Maximum allowed distance for relevant results.
        top_k (int, optional): Number of top results to retrieve. Defaults to 5.
    
    Returns:
        list of dict: Retrieved chunks with metadata.
    """
    # Get the embedding of the question
    question_embedding = get_text_embedding(question, tokenizer=tokenizer, model=embeddings_model)

    # Perform similarity search
    distances, indices = index.search(np.array([question_embedding], dtype="float32"), top_k)

    # Retrieve relevant chunks and metadata
    relevant_chunks = []
    for distance, vector_id in zip(distances[0], indices[0]):
        if vector_id != -1 and distance >= max_distance_threshold:  # Ensure ID is valid and within threshold
            for doc_hash, meta in metadata.items():
                if meta["vector_id"] == vector_id:
                    with open(f"{extracted_text_folder}{meta['doc_name']}/{meta['filename']}", "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        chunk_content = "".join(lines[2:]).strip()
                    chunk_info = {
                        "chunk": chunk_content,
                        "doc_name": meta["doc_name"],
                        "pages": meta["pages"],
                        "distance": distance  # Include distance for reference
                    }
                    relevant_chunks.append(chunk_info)
                    break

    return relevant_chunks


def ask_question(chat_client, tokenizer, embeddings_model, prompt, extracted_text_folder, index, metadata, max_distance_threshold, top_k, chat_history=[]):

    relevant_chunks = query_with_context(tokenizer, embeddings_model, prompt, extracted_text_folder, index, metadata, max_distance_threshold, top_k)
    # Display and store retrieved context
    contexts = "RAG system:\n**The following are document chunks from the database that might help you answer the user's question.**\n"
    document_names = []
    for chunk in relevant_chunks:
        contexts += f"Document name: {chunk['doc_name']}\nPages: {chunk['pages']}\nContent:\n{chunk['chunk']}\n"
        document_names.append(chunk['doc_name'])
    # print(f"Contexts:\n{contexts}")
    # print(f"Document names: {document_names}")

    messages = [
        {
            "role": "system",
            "content": f"""You are a cybersecurity expert that will answer question to users. 
            Always source information using document names and pages. 
            Pages are very important ! 
            Do not mention the dcuments from the RAG system as the user SHOULD NOT BE aware of them UNDER ANY CIRCUMSTANCES.""",
        }
    ]

    # Append chat history to messages 
    if chat_history:
        for message in chat_history[0:-1]:
            messages.append(message)

    # Append RAG documents for latest user prompt to messages 
    messages.append({
        "role": "user",
        "content": contexts
    })
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Mistral answer with context
    stream_response = chat_client.chat.stream(
        model="mistral-large-latest",
        messages = messages
    )
    for chunk in stream_response:
        yield chunk.data.choices[0].delta.content


def ask_question_and_get_sources(chat_client, tokenizer, embeddings_model, prompt, extracted_text_folder, index, metadata, max_distance_threshold, top_k, chat_history=[]):

    relevant_chunks = query_with_context(tokenizer, embeddings_model, prompt, extracted_text_folder, index, metadata, max_distance_threshold, top_k)
    # Display and store retrieved context
    contexts = ""
    document_names = []
    for chunk in relevant_chunks:
        contexts += f"RAG system:\n**The following are document chunks from the database that might help you answer the user's question.**\nDocument name: {chunk['doc_name']}\nPages: {chunk['pages']}\nContent:\n{chunk['chunk']}\n"
        document_names.append(chunk['doc_name'])
    # print(f"Contexts:\n{contexts}")
    # print(f"Document names: {document_names}")

    messages = [
        {
            "role": "system",
            "content": f"""You are a cybersecurity expert that will answer question to users. 
            Always source information using document names and pages. 
            Pages are very important ! 
            Do not mention the dcuments from the RAG system as the user SHOULD NOT BE aware of them UNDER ANY CIRCUMSTANCES.""",
        }
    ]

    # Append chat history to messages including latest user prompt
    if chat_history:
        for message in chat_history[0:-1]:
            messages.append(message)
    # Append RAG documents for latest user prompt to messages 
    for chunk in relevant_chunks:
        messages.append({
            "role": "system",
            "content": contexts
        })
    messages.append({
        "role": "user",
        "content": prompt
    })

    # Mistral answer with context
    complete_response = chat_client.chat.complete(
        model="mistral-large-latest",
        messages = messages
    )
    print(complete_response.choices[0].message.content, relevant_chunks)
    return complete_response.choices[0].message.content, relevant_chunks


def is_llm_answer_correct(chat_client, question, answer, llm_answer) -> str:
    """
    Validate if llm_answer is an appropriate response to the question by comparing it with a reference answer.

    Args:
        question (str): The question to be validated.
        answer (str): The reference answer.
        llm_answer (str): The answer generated by the LLM to validate.

    Returns:
        bool: True if the LLM answer is correct, False otherwise.
    """

    # Formulate the prompt to ask Mistral
    # prompt = f"Question: {question}\nReference Answer: {answer}\nLLM Answer: {llm_answer}\nDoes the LLM Answer appropriately respond to the question, given the Reference Answer? Be very thorough. Please respond with 'True' or 'False'."
    prompt = f"Question: {question}\nReference Answer: {answer}\nLLM Answer: {llm_answer}\nDoes the LLM Answer appropriately respond to the question, given the Reference Answer? Please respond with 'True' or 'False'."

    try:
        chat_response = chat_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # Extract Mistral's response
        response_content = chat_response.choices[0].message.content.strip()
        time.sleep(10)  # Sleep for 2 second to avoid rate limiting
        # Determine correctness based on Mistral's reply
        if response_content.lower().replace("*", "").startswith("true"):
            response_content.lower().replace("*", "").startswith("true")
            return "Correct", response_content
        elif response_content.lower().replace("*", "").startswith("false"):
            response_content.lower().replace("*", "").startswith("false")
            return "Incorrect", response_content
        else:
            return "Invalid response", response_content
    except Exception as e:
        print(f"Error while communicating with Mistral: {e}")