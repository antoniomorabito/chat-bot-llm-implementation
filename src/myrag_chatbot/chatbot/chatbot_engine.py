import os
from typing import Optional, List
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import Document
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Optional: Google Gemini
try:
    from langchain_google_genai import GenerativeModel, GoogleGenerativeAIEmbeddings
except ImportError:
    GenerativeModel = None
    GoogleGenerativeAIEmbeddings = None
    logging.warning("langchain_google_genai not found. Gemini functionality will be disabled.")

load_dotenv()  # Load environment variables

class ChatbotEngine:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_model: str = "ollama",
        temperature: float = 0.2,
        use_internet_search: bool = False
    ):
        logging.debug("Menginisialisasi ChatbotEngine...")
        self.retriever = retriever
        self.llm_model = llm_model
        self.temperature = temperature
        self.use_internet_search = use_internet_search

        self.llm = self._select_llm()
        self.internet_search = self._setup_internet_search()

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Kamu adalah chatbot cerdas. Gunakan informasi berikut untuk menjawab pertanyaan:
{context}

Jika informasi tidak mencukupi, jawab berdasarkan pengetahuan umummu.

Pertanyaan: {question}""",
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
        )

        logging.debug("ChatbotEngine berhasil diinisialisasi.")

    def _select_llm(self):
        logging.debug(f"Memilih model LLM: {self.llm_model}")
        if self.llm_model == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY harus diatur di environment variables.")
            logging.debug("Menggunakan model OpenAI (gpt-3.5-turbo)")
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=self.temperature,
                openai_api_key=openai_api_key,
            )
        elif self.llm_model == "ollama":
            logging.debug("Menggunakan model Ollama (llama3.2:latest)")
            return Ollama(model="llama3.2:latest", temperature=self.temperature)
        elif self.llm_model == "gemini":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY harus diatur di environment variables.")
            if GenerativeModel is None:
                raise ValueError("langchain_google_genai tidak tersedia.")
            logging.debug("Menggunakan model Gemini (gemini-pro)")
            return GenerativeModel(
                model_name="gemini-pro",
                temperature=self.temperature,
                api_key=google_api_key
            )
        else:
            raise ValueError(f"Model LLM tidak didukung: {self.llm_model}")

    def _setup_internet_search(self) -> Optional[TavilySearchResults]:
        if self.use_internet_search:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                logging.warning("TAVILY_API_KEY tidak ditemukan. Pencarian internet dinonaktifkan.")
                return None
            logging.debug("Pencarian internet (Tavily) diaktifkan.")
            return TavilySearchResults(max_results=3, api_key=tavily_api_key)
        logging.debug("Pencarian internet tidak diaktifkan.")
        return None

    def ask(self, question: str) -> str:
        logging.debug(f"Pertanyaan diterima: {question}")
        context_sources: List[str] = []

        # Ambil konteks dari RAG
        logging.debug("Mengambil dokumen relevan dari retriever...")
        rag_results = self.retriever.get_relevant_documents(question)
        logging.debug(f"Jumlah dokumen dari retriever: {len(rag_results)}")
        rag_context = "\n".join([doc.page_content for doc in rag_results])
        context_sources.append(f"Informasi dari dokumen:\n{rag_context}")

        # Internet Search jika aktif
        if self.use_internet_search and self.internet_search:
            try:
                logging.debug("Melakukan pencarian internet...")
                web_results = self.internet_search.run(question)
                if web_results:
                    logging.debug("Hasil pencarian internet berhasil diperoleh.")
                    context_sources.append(f"Informasi dari internet:\n{web_results}")
            except Exception as e:
                logging.error(f"Error saat pencarian internet: {e}")

        # Gabungkan semua konteks
        final_context = "\n\n".join(context_sources)
        logging.debug("Menjalankan QA Chain dengan konteks yang disiapkan...")

        result = self.qa_chain({
            "query": question,
            "input_documents": rag_results if rag_results else []
        })

        logging.debug("Jawaban berhasil diperoleh.")
        return result["result"]

    def ask_with_sources(self, question: str) -> dict:
        answer = self.ask(question)
        sources = []

        # Sumber dari dokumen
        rag_results = self.retriever.get_relevant_documents(question)
        for doc in rag_results:
            sources.append({
                "source_type": "document",
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Sumber dari internet
        if self.use_internet_search and self.internet_search:
            try:
                web_results = self.internet_search.run(question)
                if web_results:
                    sources.append({
                        "source_type": "internet_search",
                        "content": web_results,
                    })
            except Exception as e:
                logging.error(f"Error saat pencarian internet: {e}")

        return {"answer": answer, "sources": sources}
