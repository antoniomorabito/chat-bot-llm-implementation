from langchain.document_loaders import TextLoader

def load_txt(path: str):
    loader = TextLoader(path)
    return loader.load()