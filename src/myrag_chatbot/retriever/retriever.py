import os
import traceback
from typing import List

from langchain_chroma import Chroma
from chromadb.config import Settings as ClientSettings
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever

def create_retriever(
    embeddings: Embeddings,
    documents: List[Document],
    retriever_type: str = "similarity",  # "similarity", "mmr", "reranking"
    persist_directory: str = "/Users/antoniomorabito/Documents/aiproject/myrag_chatbot/src/myrag_chatbot/db",
) -> VectorStoreRetriever:
    """
    Membuat retriever dari dokumen dan embeddings.
    """
    print(f"[DEBUG] create_retriever: retriever_type={retriever_type}, persist_directory={persist_directory}")
    print(f"[DEBUG] create_retriever: type(embeddings)={type(embeddings)}, len(documents)={len(documents)}")

    try:
        client_settings = ClientSettings(
            anonymized_telemetry=False,
            persist_directory=persist_directory
        )
        print(f"[DEBUG] create_retriever: About to call Chroma with persist_directory: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"[DEBUG] create_retriever: Chroma vectorstore created successfully")
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] create_retriever: Error creating Chroma vectorstore: {e}")
        raise

    try:
        if retriever_type == "similarity":
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            print(f"[DEBUG] create_retriever: similarity retriever created")

        elif retriever_type == "mmr":
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
            )
            print(f"[DEBUG] create_retriever: mmr retriever created")

        elif retriever_type == "reranking":
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            try:
                from langchain.retrievers import EnsembleRetriever
                from langchain_community.rerankers import SentenceTransformersRerank

                reranker = SentenceTransformersRerank(top_n=3)
                retriever = EnsembleRetriever(
                    retrievers=[base_retriever],
                    weights=[1],
                    reranker=reranker
                )
                print(f"[DEBUG] create_retriever: reranking retriever created (SentenceTransformersRerank)")
            except ImportError as ie:
                print("[WARN] SentenceTransformersRerank not available, falling back to EmbeddingsFilter reranker.")
                from langchain.retrievers.document_compressors import EmbeddingsFilter
                from langchain.retrievers import ContextualCompressionRetriever

                compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                print(f"[DEBUG] create_retriever: reranking retriever created (EmbeddingsFilter fallback)")

        else:
            raise ValueError(f"Jenis retriever tidak didukung: {retriever_type}")

    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] create_retriever: Error creating retriever: {e}")
        raise

    print(f"[DEBUG] create_retriever: Retriever berhasil dibuat: {retriever}")
    return retriever
