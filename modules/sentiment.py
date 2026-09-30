"""
Module 4: Sentiment Analysis
Classifies comments as positive/negative/neutral using a pre-trained
Hugging Face model (fast, free, no API cost).
"""

from transformers import pipeline

# Model loads once (takes a bit of time the first time)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_single_comment(comment: str) -> dict:
    """
    Returns the sentiment of a single comment.
    Returns: {"label": "POSITIVE"/"NEGATIVE", "score": 0.0-1.0}
    """
    result = sentiment_pipeline(comment)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 2),
    }


def analyze_comments_batch(comments: list[str]) -> dict:
    """
    Returns an overall sentiment breakdown for multiple comments.
    Returns: percentage breakdown + overall verdict.
    """
    if not comments:
        return {
            "positive_pct": 0,
            "negative_pct": 0,
            "total_comments": 0,
            "verdict": "No comments found to analyze",
        }

    results = sentiment_pipeline(comments)

    positive_count = sum(1 for r in results if r["label"] == "POSITIVE")
    negative_count = sum(1 for r in results if r["label"] == "NEGATIVE")
    total = len(comments)

    positive_pct = round((positive_count / total) * 100, 1)
    negative_pct = round((negative_count / total) * 100, 1)

    if positive_pct >= 70:
        verdict = "Audience reaction is very positive 🎉"
    elif positive_pct >= 50:
        verdict = "Overall positive reaction, but mixed as well"
    elif negative_pct >= 50:
        verdict = "Audience reaction is largely negative ⚠️"
    else:
        verdict = "Mixed/neutral reaction"

    return {
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "total_comments": total,
        "verdict": verdict,
    }


# NOTE: To pull actual comments from YouTube you need their official
# API (YouTube Data API) - that's a separate setup.
# For now, you can manually pass a list of comments for testing.

if __name__ == "__main__":
    sample_comments = [
        "This is amazing, love it!",
        "Waste of time, not useful at all",
        "Great content, learned a lot",
        "Not what I expected",
    ]
    print(analyze_comments_batch(sample_comments))