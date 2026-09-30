"""
Module 7: Multi-Language Support
Translates the summary/bullets into any language.
No separate translation API needed - the LLM handles translation itself.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

# Languages shown in the Streamlit dropdown
SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Hinglish", "Spanish", "French",
    "German", "Tamil", "Telugu", "Bengali", "Marathi",
]


def translate_text(text: str, target_language: str) -> str:
    """
    Translates any given text (summary/bullets) into the target language.
    """
    if target_language == "English":
        return text  # skip if already in English

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a professional translator. Translate the given text into "
                   f"{target_language}. Preserve the meaning and tone, only change the language."),
        ("user", "{text}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content