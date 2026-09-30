"""
Module 1: Input Handler
Detects whether the pasted link is a valid YouTube link,
and extracts the video ID from it.
"""

import re


def extract_youtube_id(url: str) -> str | None:
    """
    Extracts the video ID from a YouTube URL.
    Example: https://youtu.be/abc123XYZ  ->  abc123XYZ
    Example: https://youtube.com/watch?v=abc123XYZ  ->  abc123XYZ
    """
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def process_link(url: str) -> dict:
    """
    Main function - this is the one called from app.py.
    Returns the video ID and validity.
    """
    url = url.strip()  # keep original case: YouTube video IDs are case-sensitive

    if "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
        return {
            "valid": False,
            "id": None,
            "message": "This doesn't look like a YouTube link. Please check again.",
        }

    video_id = extract_youtube_id(url)
    return {
        "valid": video_id is not None,
        "id": video_id,
        "message": "YouTube link detected" if video_id else "Invalid YouTube link",
    }


# Quick test - run this file directly to check
if __name__ == "__main__":
    test_links = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://random-site.com",
    ]
    for link in test_links:
        print(process_link(link))