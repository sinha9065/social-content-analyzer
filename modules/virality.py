"""
Module 5: Virality / Engagement Prediction
Predicts the viral potential of content based on features like
length, sentiment, and hook strength.

NOTE: A truly accurate prediction needs real training data
(historical views/likes data from past videos). For now this uses a
RULE-BASED SCORING SYSTEM, which is good enough for a demo/portfolio project.
If you have real data (your own or from a dataset), you can replace this
with a trained Logistic Regression/Random Forest model.
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


def score_hook_strength(first_30_sec_text: str) -> int:
    """
    Sends the text of the video's first 30 seconds to the LLM to score hook strength (1-10).
    A strong hook = viewers stay and watch = higher chance of going viral.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a social media content expert. Looking at the given opening lines, "
                   "say how strong a 'hook' it creates (its ability to stop a viewer from scrolling). "
                   "Return only a number between 1 and 10, nothing else."),
        ("user", "{opening_text}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"opening_text": first_30_sec_text})

    try:
        score = int("".join(filter(str.isdigit, response.content))[:2])
        return min(score, 10)
    except (ValueError, IndexError):
        return 5  # default if parsing fails


def calculate_virality_score(
    duration_seconds: float,
    hook_score: int,
    positive_sentiment_pct: float,
    has_trending_audio: bool = False,
) -> dict:
    """
    Rule-based scoring - combines each factor with a weight to build a final score.
    This is the same concept as your car-price-prediction project, just with
    manually set weights (you can train an ML model here if you have real data).
    """
    score = 0
    reasons = []

    # Factor 1: Duration (shorter content generally performs better, ~90 sec is ideal)
    if duration_seconds <= 90:
        score += 25
        reasons.append("Length is ideal (under 90 sec)")
    elif duration_seconds <= 180:
        score += 15
        reasons.append("Length is a bit long, but manageable")
    else:
        score += 5
        reasons.append("Length is quite long, retention may drop")

    # Factor 2: Hook strength (out of 10, weight 30)
    hook_contribution = (hook_score / 10) * 30
    score += hook_contribution
    reasons.append(f"Hook strength: {hook_score}/10")

    # Factor 3: Sentiment (out of 100%, weight 25)
    sentiment_contribution = (positive_sentiment_pct / 100) * 25
    score += sentiment_contribution
    reasons.append(f"Positive sentiment: {positive_sentiment_pct}%")

    # Factor 4: Trending audio bonus (weight 20)
    if has_trending_audio:
        score += 20
        reasons.append("Uses trending audio (+20)")
    else:
        reasons.append("No trending audio detected")

    score = round(min(score, 100), 1)

    if score >= 75:
        verdict = "High Viral Potential 🚀"
    elif score >= 50:
        verdict = "Medium Viral Potential 📈"
    else:
        verdict = "Low Viral Potential — needs improvement 📉"

    return {
        "score": score,
        "verdict": verdict,
        "reasons": reasons,
    }