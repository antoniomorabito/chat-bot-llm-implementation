# chainlit_app/app.py
import chainlit as cl
from myrag_chatbot.chatbot.chatbot_engine import ChatbotEngine

# Gunakan Ollama sebagai LLM lokal
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Placeholder retriever kosong (sementara, untuk testing chatbot saja)
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

class DummyRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        return [Document(page_content="Belum ada dokumen dimuat.")]

    async def aget_relevant_documents(self, query):
        return [Document(page_content="Belum ada dokumen dimuat.")]

# Init LLM dan dummy retriever
llm = Ollama(model="llama3.2:latest")
embedding = OllamaEmbeddings(model="llama3.2:latest")
retriever = DummyRetriever()

# Init chatbot engine
chatbot = ChatbotEngine(llm=llm, retriever=retriever)

@cl.on_message
async def handle_message(message: cl.Message):
    response = chatbot.ask(message.content)
    await cl.Message(content=response).send()
