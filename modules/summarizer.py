"""
Module 3: AI Summary Generator
Sends the transcript to an LLM to generate a summary + bullet points + timestamped sections.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)


def generate_summary(full_text: str) -> str:
    """
    Takes the full transcript and returns a crisp summary.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert content summarizer. Read the video transcript and "
                   "give a clear, 3-4 line summary that captures the main idea."),
        ("user", "Summarize this transcript:\n\n{transcript}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"transcript": full_text})
    return response.content


def generate_bullet_points(full_text: str) -> str:
    """
    Returns the video's key takeaways as bullet points.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You break content down into key takeaways. Give 5-6 bullet points "
                   "covering the most important points of the video. Keep each bullet short and clear."),
        ("user", "Extract key takeaways from this transcript:\n\n{transcript}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"transcript": full_text})
    return response.content


def generate_timestamped_summary(segments: list) -> str:
    """
    Generates a section-wise summary with timestamps.
    Example output: "0:00-0:30 -> Intro, 0:30-2:00 -> Main topic..."
    """
    # Group segments into chunks (roughly every ~60 seconds)
    chunked_text = ""
    current_chunk_start = 0
    chunk_text_parts = []

    for seg in segments:
        if seg["start"] - current_chunk_start > 60 and chunk_text_parts:
            chunked_text += f"[{int(current_chunk_start)}s - {int(seg['start'])}s]: " + " ".join(chunk_text_parts) + "\n"
            current_chunk_start = seg["start"]
            chunk_text_parts = []
        chunk_text_parts.append(seg["text"])

    if chunk_text_parts:
        chunked_text += f"[{int(current_chunk_start)}s onwards]: " + " ".join(chunk_text_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You will be given time-stamped transcript chunks. Give each chunk a short topic "
                   "label in this format: 'MM:SS - Topic name'. Return only the list, nothing else."),
        ("user", "{chunked_transcript}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"chunked_transcript": chunked_text})
    return response.content


def full_analysis(full_text: str, segments: list) -> dict:
    """
    Runs everything together - this is the function called from app.py.
    """
    return {
        "summary": generate_summary(full_text),
        "bullets": generate_bullet_points(full_text),
        "timestamped_summary": generate_timestamped_summary(segments) if segments else "N/A",
    }