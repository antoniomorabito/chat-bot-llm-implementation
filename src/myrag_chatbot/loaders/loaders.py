# src/myrag_chatbot/loaders/loaders.py
from langchain.docstore.document import Document
from myrag_chatbot.loaders.pdf_loader import load_pdf
from myrag_chatbot.loaders.txt_loader import load_txt
from typing import List

def load_documents(file_path: str) -> List[Document]:
    """
    Memuat dokumen dari file PDF atau TXT.

    Args:
        file_path: Path ke file yang akan dimuat.

    Returns:
        List berisi Document objek dari file.
    """
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".txt"):
        return load_txt(file_path)
    else:
        raise ValueError("Format file tidak didukung. Hanya PDF dan TXT yang diterima.")

if __name__ == '__main__':
    # Contoh penggunaan
    try:
        docs_pdf = load_documents("dummy.pdf")
        print(f"Berhasil memuat {len(docs_pdf)} halaman dari dummy.pdf")
    except FileNotFoundError:
        print("File dummy.pdf tidak ditemukan.")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)

    try:
        docs_txt = load_documents("dummy.txt")
        print(f"Berhasil memuat {len(docs_txt)} dokumen dari dummy.txt")
    except FileNotFoundError:
        print("File dummy.txt tidak ditemukan.")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)
