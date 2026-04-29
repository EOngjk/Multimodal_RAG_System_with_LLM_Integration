# main.py
from populate_database import load_documents_from_directory, split_documents_into_chunks, add_to_chroma
from query_data import query_rag

CHROMA_PATH = "./local_chroma_db"
DIRECTORY_PATH = r"C:/Users/user/OneDrive/Documents/Own Projects/RAG/Data Files"

def main():
    print("\nLocal RAG System Ready!")
    print("Type 'exit' to quit\n")

    documents = load_documents_from_directory(DIRECTORY_PATH)
    chunks = split_documents_into_chunks(documents)
    add_to_chroma(chunks, CHROMA_PATH)

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        response = query_rag(question, chunks, CHROMA_PATH)
        print(response)
        print("\n")

if __name__ == "__main__":
    main()