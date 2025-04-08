import chainlit as cl
import os
from myrag_chatbot.chatbot.chatbot_engine import ChatbotEngine
from myrag_chatbot.loaders.loaders import load_documents
from myrag_chatbot.splitter.splitter import split_documents
from myrag_chatbot.embedder.embedder import create_embeddings
from myrag_chatbot.retriever.retriever import create_retriever
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()
# Konfigurasi
DOCUMENT_PATH = "/Users/antoniomorabito/Documents/aiproject/myrag_chatbot/chainlit_app/black-beauty-obooko.pdf"
CHROMA_DB_PATH = "/Users/antoniomorabito/Documents/aiproject/myrag_chatbot/src/myrag_chatbot/db"
LLM_MODEL = "ollama"
EMBEDDING_MODEL = "ollama"
TEMPERATURE = 0.2
USE_INTERNET_SEARCH = True
RETRIEVER_TYPE = "reranking"
@cl.on_chat_start
async def main():
    print("[DEBUG] Memulai sesi chat...")
    try:
        # Load dokumen
        documents = load_documents(DOCUMENT_PATH)
        await cl.Message(content=f"Berhasil memuat {len(documents)} dokumen.").send()
        # Split dokumen menjadi chunk
        text_chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)
        await cl.Message(content=f"Berhasil memecah dokumen menjadi {len(text_chunks)} bagian.").send()
        # Buat embeddings
        embeddings = create_embeddings(EMBEDDING_MODEL)
        print("[DEBUG] Embeddings berhasil dibuat")
        # Buat retriever
        retriever = create_retriever(
            embeddings, text_chunks,
            retriever_type=RETRIEVER_TYPE,
            persist_directory=CHROMA_DB_PATH
        )
        print("[DEBUG] Retriever berhasil dibuat")
        # Init chatbot engine
        chatbot = ChatbotEngine(
            retriever=retriever,
            llm_model=LLM_MODEL,
            temperature=TEMPERATURE,
            use_internet_search=USE_INTERNET_SEARCH
        )
        cl.user_session.set("chatbot_engine", chatbot)
        cl.user_session.set("embeddings", embeddings)
        print("[DEBUG] Chatbot engine siap.")
    except Exception as e:
        await cl.Message(content=f"Gagal memulai chatbot: {e}").send()
        print("[ERROR]", e)
        


@cl.on_message
async def handle_message(message: cl.Message):
    chatbot_engine = cl.user_session.get("chatbot_engine")
    embeddings = cl.user_session.get("embeddings")
    if message.content and message.content.startswith("/upload"):
        # Extract the file path from the command
        file_path = message.content.split(" ")[1]
        if not os.path.exists(file_path):
            await cl.Message(content=f"File not found: {file_path}").send()
            return

        await cl.Message(content=f"Memproses file: {os.path.basename(file_path)}").send()

        try:
            # Load dan proses dokumen
            documents = load_documents(file_path)
            text_chunks = split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=text_chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )
            retriever = create_retriever(
                embeddings=embeddings,
                documents=text_chunks,
                retriever_type=RETRIEVER_TYPE,
                persist_directory=CHROMA_DB_PATH
            )
            chatbot_engine.retriever = retriever
            vectorstore.persist()
            await cl.Message(content=f"Berhasil menambahkan dokumen ke database.").send()
        except Exception as e:
            await cl.Message(content=f"Gagal memproses dokumen: {e}").send()
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)  # Remove the file after processing
    else:
        response = chatbot_engine.ask(message.content)
        await cl.Message(content=response).send()

