import streamlit as st
import pandas as pd
import pickle
import os
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import time

# Constants
VECTOR_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
THRESHOLD = 0.6

st.set_page_config(page_title="🎓 KCET FAQ Chatbot", layout="centered")

# --- Theme Toggle ---
theme = st.sidebar.radio("🌓 Select Theme", ["Dark", "Light"])
if theme == "Light":
    bg_color = "#f9f9f9"
    text_color = "#000000"
    user_bg = "#e1e1e1"
    bot_bg = "#d1d1f1"
else:
    bg_color = "#0f0f0f"
    text_color = "#ffffff"
    user_bg = "#444"
    bot_bg = "#1c1c1c"

# --- Custom CSS ---
st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    .chat-container {{
        padding: 4px 12px;
        margin-bottom: 10px;
    }}
    .user-msg, .bot-msg {{
        padding: 10px 16px;
        border-radius: 20px;
        margin: 6px 0;
        max-width: 85%;
        word-wrap: break-word;
    }}
    .user-msg {{
        background-color: {user_bg};
        color: {text_color};
        margin-left: auto;
        text-align: right;
    }}
    .bot-msg {{
        background-color: {bot_bg};
        color: {text_color};
        margin-right: auto;
        text-align: left;
    }}
    .stButton>button {{
        background-color: #555 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 16px;
    }}
    .stButton>button:hover {{
        background-color: #777 !important;
    }}
    .banner {{
        background-color: #222;
        color: #FFD700;
        font-weight: bold;
        padding: 8px;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        animation: scroll-left 20s linear infinite;
    }}
    @keyframes scroll-left {{
        0% {{ transform: translateX(100%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.markdown("""
    <div class='banner'>
        💼 100% Placement | 👩‍🏫 Top Faculty | 🧠 Hackathons | 📚 Library Access | 🛏 Hostel Facility | 🌐 Industry Collaboration
    </div>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🤖 KCET Bot Assistant</h1><hr>", unsafe_allow_html=True)

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

# --- Session State ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything.")]

# --- Display Chat ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    css_class = "user-msg" if speaker == "👤" else "bot-msg"
    st.markdown(f"<div class='{css_class}'><b>{speaker}</b>: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your message here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Clear Chat ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🤖", "👋 Hello! I'm your KCET Assistant. Ask me anything.")]
    st.rerun()

# --- Download Chat as PDF ---
if st.download_button("📥 Download Chat as PDF", data=None, file_name="kcet_chat.pdf", disabled=True, key="dummy"):
    pass  # Placeholder button
if st.session_state.chat_log:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for speaker, msg in st.session_state.chat_log:
        pdf.multi_cell(0, 10, f"{speaker}: {msg}")
    pdf_file = f"chat_{uuid.uuid4().hex}.pdf"
    pdf.output(pdf_file)
    with open(pdf_file, "rb") as f:
        st.download_button("📥 Download Chat as PDF", f, file_name="kcet_chat.pdf", mime="application/pdf")
    os.remove(pdf_file)

# --- Handle Input ---
if submitted and user_input.strip():
    user_msg = user_input.strip()
    st.session_state.chat_log.append(("👤", user_msg))

    # Vector Matching
    query_vec = vectorizer.transform([user_msg.lower()])
    similarity = cosine_similarity(query_vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    if max_sim >= THRESHOLD:
        response = df.iloc[idx]['Answer']
    else:
        response = "❌ Sorry, I couldn't understand that. Please try rephrasing."

    # Typing animation simulation
    with st.spinner("🤖 Typing..."):
        time.sleep(min(1.2, len(response) * 0.02))

    st.session_state.chat_log.append(("🤖", response))
    st.rerun()
