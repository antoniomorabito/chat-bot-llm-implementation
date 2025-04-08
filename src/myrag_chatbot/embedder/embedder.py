from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from typing import Optional
import os

def create_embeddings(embedding_model: str = "openai") -> Embeddings:
    """
    Membuat model embeddings.

    Args:
        embedding_model: Model embedding yang akan digunakan ("openai", "ollama", atau "gemini").

    Returns:
        Objek Embeddings yang sesuai.
    """
    print(f"[DEBUG] Membuat embeddings dengan model: {embedding_model}")

    if embedding_model == "openai":
        openai_api_key = os.getenv("OPEN_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY harus diatur di environment variables.")
        print("[DEBUG] Menggunakan OpenAIEmbeddings")
        return OpenAIEmbeddings(openai_api_key=openai_api_key)

    elif embedding_model == "ollama":
        print("[DEBUG] Memanggil OllamaEmbeddings...")
        try:
            # Tes langsung koneksi
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            print(f"[DEBUG] Respons manual ke Ollama: {response.status_code} - {response.text}")
            
            # Lanjut buat embeddings
            return OllamaEmbeddings(model="llama3.2:latest")
        except Exception as e:
            print(f"[ERROR] Gagal membuat OllamaEmbeddings: {e}")
            raise

    elif embedding_model == "gemini":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY harus diatur di environment variables.")
        print("[DEBUG] Menggunakan GoogleGenerativeAIEmbeddings")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)

    else:
        raise ValueError(f"Model embedding tidak didukung: {embedding_model}")

if __name__ == '__main__':
    # Contoh penggunaan
    try:
        embedding_model = create_embeddings("ollama")  # atau "openai", "gemini"
        text = "Contoh teks untuk di-embed."
        print("[DEBUG] Menghasilkan embeddings dari teks")
        embeddings = embedding_model.embed_query(text)
        print(f"[DEBUG] Panjang embeddings: {len(embeddings)}")
        print(f"[DEBUG] Contoh embeddings:\n{embeddings[:10]}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat membuat embeddings: {e}")
