"""
AI-Powered Social Media Content Analyzer
Main Streamlit App - ties together all modules.
"""

import streamlit as st
from modules import input_handler, transcript_extractor, summarizer
from modules import sentiment, virality, qa_rag, translator

st.set_page_config(page_title="AI Content Analyzer", page_icon="🎬", layout="wide")

st.title("🎬 AI-Powered Social Media Content Analyzer")
st.caption("Paste a YouTube link — get AI summary, sentiment, and virality score")

# Session state - so data doesn't get lost on page refresh
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.vector_store = None

# ---- Step 1: Link Input ----
url = st.text_input("Paste a YouTube link:")
analyze_btn = st.button("Analyze 🚀")

if analyze_btn and url:
    link_info = input_handler.process_link(url)

    if not link_info["valid"]:
        st.error(link_info["message"])
    else:
        st.success(f"{link_info['message']} ✅")

        with st.spinner("Extracting transcript..."):
            transcript_data = transcript_extractor.get_transcript(link_info["id"])

        if not transcript_data["success"]:
            st.error(f"Could not get transcript: {transcript_data.get('error', 'Unknown error')}")
        else:
            st.session_state.full_text = transcript_data["full_text"]
            st.session_state.segments = transcript_data["segments"]
            st.session_state.video_id = link_info["id"]
            st.session_state.analysis_done = True

            with st.spinner("Generating AI summary..."):
                st.session_state.analysis = summarizer.full_analysis(
                    transcript_data["full_text"], transcript_data["segments"]
                )

            with st.spinner("Processing video for Q&A..."):
                st.session_state.vector_store = qa_rag.build_vector_store(
                    transcript_data["full_text"], link_info["id"]
                )

# ---- Results Display (in Tabs) ----
if st.session_state.analysis_done:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📝 Summary", "😊 Sentiment", "🚀 Virality Score", "❓ Q&A", "🌍 Translate"]
    )

    # --- Tab 1: Summary ---
    with tab1:
        st.subheader("Summary")
        st.write(st.session_state.analysis["summary"])

        st.subheader("Key Takeaways")
        st.write(st.session_state.analysis["bullets"])

        st.subheader("Timestamped Sections")
        st.write(st.session_state.analysis["timestamped_summary"])

    # --- Tab 2: Sentiment ---
    with tab2:
        st.subheader("Comments Sentiment Analysis")
        st.info("Paste comments manually below (one comment per line) - "
                "in a real app, these would come automatically from the YouTube API")

        comments_input = st.text_area("Paste comments (one per line):", height=150)

        if st.button("Check Sentiment"):
            comments_list = [c.strip() for c in comments_input.split("\n") if c.strip()]
            with st.spinner("Analyzing..."):
                result = sentiment.analyze_comments_batch(comments_list)

            col1, col2 = st.columns(2)
            col1.metric("Positive", f"{result['positive_pct']}%")
            col2.metric("Negative", f"{result['negative_pct']}%")
            st.write(f"**Verdict:** {result['verdict']}")

    # --- Tab 3: Virality Score ---
    with tab3:
        st.subheader("Viral Potential Prediction")

        duration = st.number_input("Video length (in seconds):", min_value=1, value=60)
        has_trending_audio = st.checkbox("Uses trending audio?")

        if st.button("Calculate Virality Score"):
            first_30_sec_text = st.session_state.full_text[:300]  # approx first 30 sec

            with st.spinner("Checking hook strength..."):
                hook_score = virality.score_hook_strength(first_30_sec_text)

            # Sentiment wasn't run, so defaulting to 60% for this demo
            result = virality.calculate_virality_score(
                duration_seconds=duration,
                hook_score=hook_score,
                positive_sentiment_pct=60,
                has_trending_audio=has_trending_audio,
            )

            st.metric("Virality Score", f"{result['score']}/100")
            st.write(f"**{result['verdict']}**")
            st.write("**Breakdown:**")
            for reason in result["reasons"]:
                st.write(f"- {reason}")

    # --- Tab 4: Q&A ---
    with tab4:
        st.subheader("Ask Questions About the Video")
        question = st.text_input("Your question:")

        if st.button("Get Answer") and question:
            with st.spinner("Searching..."):
                answer = qa_rag.ask_question(st.session_state.vector_store, question)
            st.write(f"**Answer:** {answer}")

    # --- Tab 5: Translate ---
    with tab5:
        st.subheader("Translate Summary")
        target_lang = st.selectbox("Choose language:", translator.SUPPORTED_LANGUAGES)

        if st.button("Translate"):
            with st.spinner("Translating..."):
                translated = translator.translate_text(
                    st.session_state.analysis["summary"], target_lang
                )
            st.write(translated)