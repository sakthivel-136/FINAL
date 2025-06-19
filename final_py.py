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
        margin: 0 auto 20px auto;
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
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 720px;
        z-index: 999;
        background: #0f0f0f;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Marquee Banner ---
st.markdown("<div class='marquee'>💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Connect</div>", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 style='text-align:center;'>🤖 KCET Bot Assistant</h1><hr>", unsafe_allow_html=True)

# --- Load Pickle File ---
@st.cache_data
def load_pickle():
    if not os.path.exists(VECTOR_FILE):
        st.error("❌ Required vector file not found. Please generate 'vectorized (3).pkl'.")
        st.stop()
    with open(VECTOR_FILE, "rb") as f:
        vectorizer, vectors, df = pickle.load(f)
    return vectorizer, vectors, df

vectorizer, vectors, df = load_pickle()

# --- Session State Setup ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]

# --- Clean old audio files ---
for file in glob.glob("tts_output_*.mp3"):
    os.remove(file)

# --- Chat Display FIRST ---
chat_display = st.empty()
with chat_display.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for speaker, msg in st.session_state.chat_log:
        css_class = "user-msg" if speaker == "👤" else "bot-msg"
        st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input Box at Bottom ---
st.markdown("<div class='input-area'>", unsafe_allow_html=True)
with st.form(key="input_form", clear_on_submit=True):
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

        st.session_state.chat_log.append(("🤖", answer))

        # TTS audio
        audio_file = f"tts_output_{uuid.uuid4().hex}.mp3"
        gTTS(text=answer).save(audio_file)
        st.markdown(f"""
        <audio autoplay="true">
          <source src="{audio_file}" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

        # Refresh
        st.rerun()

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]
    st.rerun()
