# src/myrag_chatbot/loaders/txt_loader.py
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from typing import List

def load_txt(file_path: str) -> List[Document]:
    """Memuat dokumen dari file TXT."""
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"File TXT tidak ditemukan di: {file_path}")
    except Exception as e:
        raise Exception(f"Terjadi kesalahan saat memuat file TXT: {e}")

if __name__ == '__main__':
    # Contoh penggunaan (buat file dummy.txt untuk testing)
    try:
        with open("dummy.txt", "w") as f:
            f.write("Ini adalah baris pertama dari file teks.\nIni adalah baris kedua yang lebih panjang.")
        docs = load_txt("dummy.txt")
        print(f"Berhasil memuat {len(docs)} dokumen dari dummy.txt")
        if docs:
            print(f"Contoh konten dokumen pertama:\n{docs[0].page_content}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)
