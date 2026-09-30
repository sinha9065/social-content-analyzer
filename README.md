# 🎬 AI-Powered Social Media Content Analyzer
**Live Demo:**[https://social-content-analyzer-mubhuafw2pwump2gwtjatd.streamlit.app/]

An AI-powered analysis tool for YouTube videos — get an AI summary, sentiment breakdown, viral potential score, and a Q&A assistant, all in one Streamlit app.

Paste a YouTube link and get:
- A clear summary and key takeaways
- A timestamped breakdown of the video's sections
- Sentiment analysis on comments (positive/negative)
- A viral potential score based on length, hook strength, and trending-audio use
- A Q&A assistant that answers questions using only the video's own transcript (RAG)
- A summary translated into the language of your choice

## Features

| Feature | How it works |
|---|---|
| Transcript extraction | Pulls the YouTube captions directly, with timestamps |
| AI summary | LangChain + Groq (LLM) generates a summary, bullet-point takeaways, and timestamped sections |
| Sentiment analysis | A Hugging Face DistilBERT model classifies pasted comments as positive/negative |
| Virality score | A rule-based scoring model combining video length, hook strength (LLM-scored), and trending-audio use |
| Q&A (RAG) | The transcript is chunked, embedded, and stored in ChromaDB; questions are answered strictly from the retrieved transcript context, so the assistant says so instead of guessing when the answer isn't in the video |
| Translation | The summary can be translated into Hindi, Hinglish, Spanish, French, Tamil, and more |

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-DistilBERT-EE4C2C?logo=pytorch&logoColor=white)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-Embeddings-lightgrey)

## Project Structure

```
social-content-analyzer/
├── app.py                       # Main Streamlit app
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── modules/
    ├── __init__.py
    ├── input_handler.py         # Validates YouTube links, extracts video ID
    ├── transcript_extractor.py  # Extracts transcript + timestamps from a video
    ├── summarizer.py            # AI summary, bullet points, timestamped sections
    ├── sentiment.py             # Comment sentiment analysis
    ├── virality.py               # Viral potential scoring
    ├── qa_rag.py                  # RAG-based Q&A over the transcript
    └── translator.py              # Multi-language summary translation
```

## Setup

1. Clone the repo and move into the folder:
   ```bash
   git clone <your-repo-url>
   cd social-content-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

3. Install PyTorch first (the CPU build avoids a common Windows DLL error you'd otherwise hit with the default PyPI package), then install everything else:
   ```bash
   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your Groq API key (free at [console.groq.com](https://console.groq.com)):
   ```
   GROQ_API_KEY=your_key_here
   ```

5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes

- Only YouTube links are supported. A video needs captions (auto-generated or manual) available for the transcript to be extracted.
- The Q&A assistant is deliberately restricted to the video's own transcript — if you ask something the video doesn't cover, it will say so rather than answering from general knowledge.
- The virality score is a rule-based heuristic for demonstration purposes, not a model trained on real engagement data.
