import gc
import os
import shutil
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_embedding_function import get_embedding_function


def split_documents_into_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def load_documents_from_directory(directory_path):
    all_docs = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            loader = PyMuPDFLoader(
                file_path,
                extract_images=True,
                images_parser=TesseractBlobParser()
            )
            all_docs.extend(loader.load())

    return all_docs


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}_{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks, chroma_path):
    chunks.sort(key=lambda x: (x.metadata.get("source", ""), x.metadata.get("page", 0)))
    db = None

    try:
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=get_embedding_function()
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if new_chunks:
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

    finally:
        if db is not None:
            del db
        gc.collect()
        time.sleep(1)


def clear_database(chroma_path):
    gc.collect()
    time.sleep(1)

    if os.path.exists(chroma_path):
        print("🗑️ Clearing database...")

        for attempt in range(5):
            try:
                shutil.rmtree(chroma_path)
                print("✅ Database cleared.")
                break
            except PermissionError as e:
                print(f"⚠️ Attempt {attempt + 1}: DB still in use: {e}")
                gc.collect()
                time.sleep(2)
        else:
            print("❌ Failed to delete database after multiple retries.")
            print("Tip: Restart kernel / close notebook and try again.")