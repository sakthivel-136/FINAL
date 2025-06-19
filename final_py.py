

import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import smtplib
from email.message import EmailMessage
import re

# --- Constants ---
VECTOR_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
THRESHOLD = 0.6
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"  # Replace with your email
SENDER_PASSWORD = "qwertyuiop123-"  # Replace with your app password

# --- Page Setup ---
st.set_page_config(page_title="KCET Chatbot", layout="centered")

# --- Scroll Banner ---
st.markdown("""
    <style>
    .scrolling-banner {
        overflow: hidden;
        white-space: nowrap;
        box-sizing: border-box;
        animation: scroll-left 20s linear infinite;
        color: gold;
        background-color: #111;
        padding: 8px;
        font-weight: bold;
        font-size: 16px;
        text-align: center;
    }

    @keyframes scroll-left {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    <div class="scrolling-banner">
        💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
    </div>
""", unsafe_allow_html=True)

# --- Load Data ---
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

# --- Chat History ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "Hello! I'm your KCET Assistant. Ask me anything.")]

# --- Display Chat ---
st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    align = 'right' if speaker == '👤' else 'left'
    bg = '#444' if speaker == '👤' else '#222'
    st.markdown(f"<div style='background-color:{bg}; padding:10px; border-radius:10px; text-align:{align}; margin:5px 0;'>"
                f"<b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your question here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Clear Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🤖", "Hello! I'm your KCET Assistant. Ask me anything.")]
    st.rerun()

# --- Chat Logic ---
if submitted and user_input.strip():
    user_msg = user_input.strip()
    st.session_state.chat_log.append(("👤", user_msg))

    vec = vectorizer.transform([user_msg.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    if max_sim >= THRESHOLD:
        bot_msg = df.iloc[idx]['Answer']
    else:
        bot_msg = "❌ Sorry, I couldn't understand that. Please rephrase."

    with st.spinner("🤖 Typing..."):
        time.sleep(min(1.5, len(bot_msg) * 0.02))
    st.session_state.chat_log.append(("🤖", bot_msg))
    st.rerun()

# --- PDF Export Section ---
st.markdown("---")
st.markdown("### 📥 Export Chat")
email = st.text_input("Enter your email to receive the chat log (PDF):")

if st.button("Send PDF to Email"):
    if not email or "@" not in email:
        st.error("⚠️ Please enter a valid email address.")
    else:
        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)

        def clean_text(text):
            return re.sub(r'[^
