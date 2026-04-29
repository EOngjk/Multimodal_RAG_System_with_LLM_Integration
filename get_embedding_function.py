from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )