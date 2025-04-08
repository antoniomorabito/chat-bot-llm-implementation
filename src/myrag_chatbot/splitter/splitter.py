# src/myrag_chatbot/splitter/splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Memecah dokumen menjadi chunk.

    Args:
        documents: List dokumen yang akan dipecah.
        chunk_size: Ukuran chunk.
        chunk_overlap: Overlap antar chunk.

    Returns:
        List berisi chunk dokumen.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == '__main__':
    # Contoh penggunaan
    dummy_documents = [
        Document(page_content="Ini adalah bagian pertama dari dokumen yang sangat panjang. " * 50),
        Document(page_content="Ini adalah bagian kedua dari dokumen yang juga sangat panjang. " * 50),
    ]
    chunks = split_documents(dummy_documents)
    print(f"Berhasil memecah menjadi {len(chunks)} chunk.")
    print(f"Contoh chunk pertama:\n{chunks[0].page_content[:100]}...")
