import streamlit as st
import pandas as pd
import pickle
import os
import time
import uuid
import glob
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
VECTOR_FILE = "vectorized (3).pkl"
THRESHOLD = 0.6

# Page Config
st.set_page_config(page_title="🎓 KCET FAQ Chatbot", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background-color: #0f0f0f;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .marquee {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        animation: marquee 15s linear infinite;
        color: #ffcc00;
        font-weight: bold;
        padding: 10px 0;
        font-size: 18px;
        background: #111;
        text-align: center;
    }
    @keyframes marquee {
        0%   { transform: translateX(100%) }
        100% { transform: translateX(-100%) }
    }
    .chat-container {
        max-width: 720px;
        margin: 0 auto;
        height: 500px;
        overflow-y: auto;
        padding: 10px 20px;
        background: #0f0f0f;
        border: 1px solid #222;
        border-radius: 12px;
    }
    .user-msg, .bot-msg {
        padding: 12px 16px;
        border-radius: 20px;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-msg {
        background-color: #333;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .bot-msg {
        background-color: #111;
        color: white;
        margin-right: auto;
        text-align: left;
    }
    .stButton>button {
        background-color: #444 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #666 !important;
    }
    .input-area {
        max-width: 720px;
        margin: 20px auto 0;
    }
    </style>
""", unsafe_allow_html=True)

# Marquee Banner
st.markdown("<div class='marquee'>💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Connect</div>", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align:center;'>🤖 KCET Bot Assistant</h1><hr>", unsafe_allow_html=True)

# Load vectorizer and dataset
@st.cache_data
def load_pickle():
    if not os.path.exists(VECTOR_FILE):
        st.error("❌ Required vector file not found. Please generate 'vectorized (3).pkl'.")
        st.stop()
    with open(VECTOR_FILE, "rb") as f:
        vectorizer, vectors, df = pickle.load(f)
    return vectorizer, vectors, df

vectorizer, vectors, df = load_pickle()

# Session state
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]

# Clean up old audio
for file in glob.glob("tts_output_*.mp3"):
    os.remove(file)

# Display chat
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    css_class = "user-msg" if speaker == "👤" else "bot-msg"
    st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input Area ---
st.markdown("<div class='input-area'>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    with col1:
        user_input = st.text_input("Type your question here...", key="chat_input", label_visibility="collapsed")
    with col2:
        send_clicked = st.form_submit_button("➤")
st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Logic ---
if send_clicked and user_input.strip():
    query = user_input.strip().lower()
    st.session_state.chat_log.append(("👤", user_input))
    try:
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, vectors)
        max_sim = similarity.max()
        max_index = similarity.argmax()

        if max_sim >= THRESHOLD:
            answer = df.iloc[max_index]["Answer"]
        else:
            answer = "❌ Sorry, I couldn't understand that. Please try rephrasing."

        # Typing effect
        typing_placeholder = st.empty()
        typed_text = ""
        for char in answer:
            typed_text += char
            typing_placeholder.markdown(f"<div class='bot-msg'><b>🤖</b>: {typed_text}</div>", unsafe_allow_html=True)
            time.sleep(0.015)

        # Save to chat log
        st.session_state.chat_log.append(("🤖", answer))

        # Voice
        audio_file = f"tts_output_{uuid.uuid4().hex}.mp3"
        tts = gTTS(answer)
        tts.save(audio_file)

        st.markdown(f"""
        <audio autoplay="true">
          <source src="{audio_file}" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# --- Clear Chat ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]
    st.experimental_rerun()
