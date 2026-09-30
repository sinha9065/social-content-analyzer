"""
Module 6: Q&A via RAG (Retrieval Augmented Generation)
Stores the transcript in a vector database, then retrieves relevant
chunks to answer the user's question.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


def build_vector_store(full_text: str, video_id: str):
    """
    Splits the transcript into chunks, generates embeddings, and stores them in Chroma.
    Uses a separate collection per video (named using video_id).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(full_text)

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=f"video_{video_id}",
        persist_directory="./chroma_db",
    )
    return vector_store


def ask_question(vector_store, question: str) -> str:
    """
    Takes the user's question, retrieves relevant chunks, then generates an answer using the LLM.
    """
    # Find the top 3 most relevant chunks
    relevant_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You answer questions about a video using ONLY the transcript excerpts provided.\n"
         "Rules:\n"
         "1. Use only information that is explicitly present in the excerpts.\n"
         "2. Do NOT use outside knowledge, even if you know the answer. Do not add "
         "names, dates, or facts that are not in the excerpts.\n"
         "3. If the excerpts do not contain the answer, reply with exactly: "
         "'This information is not present in the video.'\n"
         "4. If the excerpts only partly answer the question, answer only the part "
         "they cover and say the rest is not mentioned in the video.\n"
         "5. Keep the answer short (2-4 sentences)."),
        ("user", "Transcript excerpts:\n\"\"\"\n{context}\n\"\"\"\n\nQuestion: {question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# Full flow - these 2 functions are used from app.py:
# 1. build_vector_store() - called once when a video loads
# 2. ask_question() - called every time the user asks a question