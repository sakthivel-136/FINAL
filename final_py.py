import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    .chat-container {
        padding: 4px 12px;
        margin-bottom: 10px;
    }
    .user-msg, .bot-msg {
        padding: 10px 16px;
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
        margin-top: 4px;
    }
    .stButton>button:hover {
        background-color: #777 !important;
    }
    .banner {
        background-color: #111;
        color: #FFD700;
        font-weight: bold;
        padding: 8px;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        animation: scroll-left 20s linear infinite;
    }
    @keyframes scroll-left {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.markdown("""
    <div class='banner'>
        💼 100% Placement | 👩‍🏫 Experienced Faculty | 🧪 Research Labs | 🧠 Hackathons | 🤝 Industry Collaboration | 🌐 Autonomous Institution
    </div>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🤖 KCET Bot Assistant</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

# --- Load Vectorizer & Data ---
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

# --- Session State ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything about the college or exams.")]

# --- Display Chat ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    css_class = "user-msg" if speaker == "👤" else "bot-msg"
    st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input Area at Bottom ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your question here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = []
    st.rerun()

# --- Handle Input Safely ---
if submitted and user_input.strip():
    st.session_state.pending_input = user_input.strip()

# --- Process After UI Renders ---
if "pending_input" in st.session_state:
    user_msg = st.session_state.pending_input
    del st.session_state.pending_input

    # Add user question
    st.session_state.chat_log.append(("👤", user_msg))

    # Vector search
    query_vec = vectorizer.transform([user_msg.lower()])
    similarity = cosine_similarity(query_vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    if max_sim >= THRESHOLD:
        response = df.iloc[idx]['Answer']
    else:
        response = "❌ Sorry, I couldn't understand that. Please rephrase your question."

    # Add bot response
    st.session_state.chat_log.append(("🤖", response))

    # Show updated chat
    st.rerun()
