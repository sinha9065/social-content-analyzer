"""
Module 2: Transcript Extraction
Extracts the transcript of a YouTube video, along with timestamps,
using YouTube's own captions API (fast, free).
"""

from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript(video_id: str) -> dict:
    """
    Extracts the transcript of a YouTube video along with timestamps.
    Returns: {"success": bool, "segments": [...], "full_text": str}
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en", "hi"])

        segments = []
        full_text_parts = []

        for entry in transcript_list:
            segments.append({
                "start": entry.start,           # start time in seconds
                "duration": entry.duration,
                "text": entry.text,
            })
            full_text_parts.append(entry.text)

        return {
            "success": True,
            "segments": segments,
            "full_text": " ".join(full_text_parts),
        }

    except Exception as e:
        return {
            "success": False,
            "segments": [],
            "full_text": "",
            "error": str(e),
        }


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds into "MM:SS" format.
    Example: 150.5 -> "2:30"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"