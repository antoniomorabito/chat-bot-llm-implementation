# src/myrag_chatbot/loaders/pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from typing import List

def load_pdf(file_path: str) -> List[Document]:
    """Memuat dokumen dari file PDF."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"File PDF tidak ditemukan di: {file_path}")
    except Exception as e:
        raise Exception(f"Terjadi kesalahan saat memuat file PDF: {e}")

if __name__ == '__main__':
    # Contoh penggunaan (buat file dummy.pdf untuk testing)
    try:
        docs = load_pdf("dummy.pdf")
        print(f"Berhasil memuat {len(docs)} halaman dari dummy.pdf")
        if docs:
            print(f"Contoh konten halaman pertama:\n{docs[0].page_content[:200]}...")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)
