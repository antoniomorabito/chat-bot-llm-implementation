# src/myrag_chatbot/chatbot/chatbot_engine.py

from typing import Optional
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun

class ChatbotEngine:
    def __init__(self, llm, retriever, internet_search: Optional[object] = None):
        self.llm = llm
        self.retriever = retriever
        self.internet_search = internet_search

        # Prompt khusus agar bot menjawab berdasarkan cerita
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Kamu adalah chatbot cerdas yang telah membaca cerita berikut:
            {context}

            Berdasarkan cerita di atas, jawab pertanyaan berikut:
            {question}
            """
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )

    def ask(self, question: str, use_internet: bool = False) -> str:
        if use_internet and self.internet_search:
            web_context = self.internet_search.run_search(question)
            question += f"\nKonteks dari internet:\n{web_context}"

        return self.qa_chain.run(question)

    def ask_with_sources(self, question: str) -> dict:
        result = self.qa_chain({"query": question}, return_only_outputs=False)
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }
