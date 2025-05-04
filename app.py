import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the provided information and formulate a comprehensive, well-structured response to the question.

The context will be passed as "Context:"
The user's question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive and covers all relevant aspects found in the context.
5. If the context does not contain sufficient information to fully answer the question, clearly state this in your response.

Format your response as follows:
1. Use clear and concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""



def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """
    Processes an uploaded PDF file by converting it into text chunks.

    This function saves the uploaded PDF file temporarily, then reads and splits
    its content into smaller text chunks using a recursive character splitting strategy.

    Args:
        uploaded_file: A Streamlit UploadedFile object representing the uploaded PDF file.

    Returns:
        A list of Document objects, each containing a chunk of text extracted from the PDF.

    Raises:
        IOError: If there is an issue reading from or writing to the temporary file.
    """
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    """
    Gets or creates a ChromaDB collection for vector storage.

    Initializes an Ollama embedding function using the `nomic-embed-text` model and 
    sets up a persistent ChromaDB client. Returns a collection configured for storing 
    and querying document embeddings using cosine similarity.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding 
        function and cosine similarity for vector operations.
    """

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./smart_qna_system")
    return chroma_client.get_or_create_collection(
        name="PDF_Explore",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """
    Adds document splits to a vector collection for semantic search.

    This function takes a list of document splits and adds them to a ChromaDB vector 
    collection, including associated metadata and unique IDs derived from the filename.

    Args:
        all_splits (list[Document]): A list of Document objects containing text chunks and metadata.
        file_name (str): A string identifier used to generate unique IDs for the text chunks.

    Returns:
        None. Displays a success message via Streamlit upon completion.

    Raises:
        ChromaDBError: If there is an issue upserting documents into the collection.
    """

    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("🎉 Success! Your document data has been added to the vector store. Ready for fast and efficient semantic search! 🚀")



def query_collection(prompt: str, n_results: int = 10):
    """
    Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt (str): The search query used to find relevant documents.
        n_results (int, optional): The maximum number of results to return. Defaults to 10.

    Returns:
        dict: A dictionary containing the query results, including documents, distances, and metadata from the collection.

    Raises:
        ChromaDBError: If there is an issue querying the collection.
    """

    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str):
    """
    Calls the language model with context and a prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and 
    a question prompt. The model applies a system prompt to appropriately format and 
    ground its responses.

    Args:
        context (str): A string containing the relevant context to answer the question.
        prompt (str): A string containing the user's question.

    Yields:
        str: Chunks of the generated response as they become available from the model.

    Raises:
        OllamaError: If there are issues communicating with the Ollama API.
    """

    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """
    Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    This function uses the MS MARCO MiniLM cross-encoder model to re-rank the input 
    documents based on their relevance to the query prompt. It returns the concatenated 
    text of the top 3 most relevant documents along with their indices.

    Args:
        documents (list[str]): A list of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents.
            - relevant_text_ids (list[int]): List of indices for the top-ranked documents.

    Raises:
        ValueError: If the documents list is empty.
        RuntimeError: If the cross-encoder model fails to load or rank documents.
    """

    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="🌟 RAG QnA: Instant PDF Insights & Answers! 🚀")
        uploaded_file = st.file_uploader(
            "**🚀 Upload Your PDF & Get Instant QnA Insights!**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "🔍 Explore & Unlock Answers! 🗝️",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("🔍 Unlock Insights from Your Document with RAG QA")
    prompt = st.text_area("**What question do you have about your document? Dive in and discover!**")
    ask = st.button(
        "🚀 Get Answers",
    )

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("📜 View Retrieved Documentss"):
            st.write(results)

        with st.expander("🏆 See Top Relevant Documents & IDs"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
