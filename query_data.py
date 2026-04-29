import gc
import time

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from get_embedding_function import get_embedding_function


llm = OllamaLLM(model="llama3.2:3b")

prompt = ChatPromptTemplate.from_template("""
You are an expert AI Assistant specializing in the analysing provided documents and answering questions based on the retrieved context.

### CONTEXT:
{context}

### QUESTION:
{question}

### INSTRUCTIONS:
1. USE ONLY the provided 'Context' to answer the 'Question'.
2. IF the answer is not contained within the Context, explicitly state: "I do not have enough information in the provided documents to answer this."
3. DO NOT use outside knowledge or make up facts.
4. Answer concisely and use a professional tone.

### HELPFUL ANSWER:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_hybrid_retriever(chunks, db: Chroma):
    vector_retriever = db.as_retriever(search_kwargs={"k": 10})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    compressor = FlashrankRerank(top_n=3)

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return final_retriever


def query_rag(query_text: str, chunks, chroma_path: str):
    db = None

    try:
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=get_embedding_function()
        )

        retriever = get_hybrid_retriever(chunks, db)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response_text = rag_chain.invoke(query_text)

        relevant_docs = retriever.invoke(query_text)
        sources = [doc.metadata.get("id", "Unknown") for doc in relevant_docs]

        formatted_response = f"Question: {query_text}\nResponse: {response_text}\nSources: {sources}"
        print(formatted_response)

        return response_text

    finally:
        if db is not None:
            del db
        gc.collect()
        time.sleep(1)