import streamlit as st
import pandas as pd
import pickle
import os
import time
import uuid
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
VECTOR_FILE = "vectorized (3).pkl"
THRESHOLD = 0.6

st.set_page_config(page_title="🎓 KCET FAQ Chatbot", layout="centered")

# --- Custom CSS & JS ---
st.markdown("""
    <style>
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
        max-width: 700px;
        margin: 0 auto;
        height: 500px;
        overflow-y: auto;
        padding-bottom: 10px;
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
    </style>
    <script>
    function scrollToBottom() {
        var container = window.parent.document.querySelector('.chat-container');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }
    window.addEventListener('load', scrollToBottom);
    </script>
""", unsafe_allow_html=True)

st.markdown("<div class='marquee'>💼 100% Placement Assistance | 👩‍🏫 Well-trained Faculty | 🎓 Industry-ready Curriculum | 🧠 Hackathons & Internships</div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🤖 KCET Bot Assistant</h1><hr>", unsafe_allow_html=True)

# Load vector data
@st.cache_data
def load_pickle():
    if not os.path.exists(VECTOR_FILE):
        st.error("❌ vectorized (3).pkl not found.")
        st.stop()
    with open(VECTOR_FILE, "rb") as f:
        vectorizer, vectors, df = pickle.load(f)
    return vectorizer, vectors, df

vectorizer, vectors, df = load_pickle()

# Session state
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "👋 Vanakkam! Ask me anything about KCET.")]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if "language" not in st.session_state:
    st.session_state.language = "English"

# Clear previous input if flagged
if st.session_state.clear_input:
    st.session_state.chat_input = ""
    st.session_state.clear_input = False

# Language selector
st.session_state.language = st.radio("🗣️ Choose your answer language:", ["English", "Tamil"], horizontal=True)

# Chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    css_class = "user-msg" if speaker == "👤" else "bot-msg"
    st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Text input + submit
st.markdown(f"""
<form onsubmit="document.getElementById('send-button').click(); return false;">
<input type="text" name="message" placeholder="Type your question here..." 
       id="chat-box" style="width: 90%; padding: 10px; border-radius: 8px; 
       background: #1e1e1e; color: white; border: 1px solid #444;" 
       onkeydown="if(event.key==='Enter'){ event.preventDefault(); this.form.submit(); }" 
       value="{st.session_state.chat_input}">
<button id="send-button" hidden></button>
</form>
""", unsafe_allow_html=True)

send_clicked = st.button("➤", key="send_button")

# Chat logic
user_input = st.session_state.chat_input
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

        # If Tamil selected, translate
        if st.session_state.language == "Tamil":
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='en', target='ta').translate(answer)
        else:
            translated = answer

        # Typing effect
        typing_placeholder = st.empty()
        typed_text = ""
        for char in translated:
            typed_text += char
            typing_placeholder.markdown(f"<div class='bot-msg'><b>🤖</b>: {typed_text}</div>", unsafe_allow_html=True)
            time.sleep(0.015)

        st.session_state.chat_log.append(("🤖", translated))
        st.session_state.clear_input = True

        # gTTS audio
        audio_file = f"tts_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=translated, lang='ta' if st.session_state.language == "Tamil" else 'en')
        tts.save(audio_file)

        # Autoplay
        st.markdown(f"""
        <audio autoplay="true">
          <source src="{audio_file}" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🤖", "👋 Vanakkam! Ask me anything about KCET.")]
    st.session_state.chat_input = ""
