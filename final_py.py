import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr

# Constants
VECTOR_FILE = "vectorized (3).pkl"
CSV_FILE = "kcet.csv"
THRESHOLD = 0.6

# Page Config
st.set_page_config(page_title="🎓 KCET FAQ Chatbot", layout="centered")

# Custom CSS & Banner
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
        box-sizing: border-box;
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
        border: none;
        padding: 10px 16px;
        font-size: 18px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #666 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Banner
st.markdown("<div class='marquee'>💼 100% Placement Assistance | 👩‍🏫 Well-trained Faculty | 🎓 Industry-ready Curriculum | 🧠 Hackathons & Internships</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>🤖 KCET Bot Assistant</h1><hr>", unsafe_allow_html=True)

# Load or generate vectorizer
@st.cache_data
def load_or_vectorize():
    if os.path.exists(VECTOR_FILE):
        with open(VECTOR_FILE, "rb") as f:
            vectorizer, vectors, df = pickle.load(f)
    else:
        if not os.path.exists(CSV_FILE):
            st.error("❌ Data file 'kcet.csv' not found!")
            st.stop()
        df = pd.read_csv(CSV_FILE)
        df['Question'] = df['Question'].str.strip().str.lower()
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(df['Question'])
        with open(VECTOR_FILE, "wb") as f:
            pickle.dump((vectorizer, vectors, df), f)
    return vectorizer, vectors, df

vectorizer, vectors, df = load_or_vectorize()

# Session states
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# TTS function
def speak(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    st.audio(mp3_fp.getvalue(), format="audio/mp3")

# Speech Recognition
def record_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"✅ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("❌ Could not understand audio.")
        except sr.RequestError:
            st.error("❌ Could not request results.")
    return ""

# Chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    css_class = "user-msg" if speaker == "👤" else "bot-msg"
    st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll to bottom
st.markdown("""
    <script>
        var body = window.parent.document.querySelector(".main");
        body.scrollTo(0, body.scrollHeight);
    </script>
""", unsafe_allow_html=True)

# Voice input button
if st.button("🎙️ Speak"):
    spoken = record_voice()
    if spoken:
        st.session_state.chat_input = spoken

# Input with form (Enter-to-send + button)
with st.form("chat_form", clear_on_submit=False):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("Type your question here...", label_visibility="collapsed", key="chat_input")
    with col2:
        submitted = st.form_submit_button("➤")

if submitted and user_input.strip():
    query = user_input.strip().lower()
    try:
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, vectors)
        max_sim = similarity.max()
        max_index = similarity.argmax()

        if max_sim >= THRESHOLD:
            answer = df.iloc[max_index]["Answer"]
        else:
            answer = "❌ Sorry, I couldn't understand that. Please try rephrasing."

        st.session_state.chat_log.append(("👤", user_input))
        st.session_state.chat_log.append(("🤖", answer))
        speak(answer)  # TTS playback
        st.session_state.chat_input = ""  # Clear input
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]
    st.session_state.chat_input = ""
