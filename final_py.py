import streamlit as st
import pandas as pd
import pickle
import os
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import uuid

# Constants
VECTOR_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
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
    .banner {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        box-sizing: border-box;
        background-color: #111;
        color: #FFD700;
        font-weight: bold;
        padding: 10px 0;
    }
    .banner-content {
        display: inline-block;
        padding-left: 100%;
        animation: scroll-left 15s linear infinite;
    }
    @keyframes scroll-left {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
    }
    .chat-container {
        padding: 4px 16px;
        margin-top: 10px;
    }
    .user-msg, .bot-msg {
        padding: 10px 14px;
        border-radius: 20px;
        margin: 6px 0;
        max-width: 85%;
        word-wrap: break-word;
    }
    .user-msg {
        background-color: #444;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .bot-msg {
        background-color: #1c1c1c;
        color: white;
        margin-right: auto;
        text-align: left;
    }
    .stButton>button {
        background-color: #555 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 16px;
        margin-top: 8px;
    }
    .stButton>button:hover {
        background-color: #777 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.markdown("""
    <div class='banner'>
        <div class='banner-content'>
            💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaborations
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>😁 KCET Bot Assistant</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

# --- Load vectorizer and data ---
@st.cache_data
def load_vector_data():
    if os.path.exists(VECTOR_FILE):
        with open(VECTOR_FILE, "rb") as f:
            vectorizer, vectors, df = pickle.load(f)
    else:
        df = pd.read_csv(CSV_FILE)
        df['Question'] = df['Question'].str.lower().str.strip()
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(df['Question'])
        with open(VECTOR_FILE, "wb") as f:
            pickle.dump((vectorizer, vectors, df), f)
    return vectorizer, vectors, df

vectorizer, vectors, df = load_vector_data()

# --- Session state ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "📅 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]

# --- Display chat log ---
if st.session_state.chat_log:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for speaker, msg in st.session_state.chat_log:
        css_class = "user-msg" if speaker == "👤" else "bot-msg"
        st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Input form (BOTTOM of page) ---
with st.form("chat_input_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Ask something...", key="user_input", label_visibility="collapsed")
    send = col2.form_submit_button("➤")

# --- Clear Chat ---
if st.button("🪯 Clear Chat"):
    st.session_state.chat_log = []
    st.rerun()

# --- Handle message ---
if send and user_input:
    query = user_input.strip().lower()
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    if max_sim >= THRESHOLD:
        response = df.iloc[idx]['Answer']
    else:
        response = "❌ Sorry, I couldn't understand that. Please rephrase."

    # Save to chat log
    st.session_state.chat_log.append(("👤", user_input))
    st.session_state.chat_log.append(("🤖", response))

    # --- TTS Audio ---
    try:
        tts = gTTS(text=response, lang='en')
        audio_file = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)
        with open(audio_file, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)
        os.remove(audio_file)
    except Exception as e:
        st.error(f"TTS Error: {e}")
