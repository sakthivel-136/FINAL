import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import time
import base64
import smtplib
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from gtts import gTTS

# --- Constants ---
VECTOR_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
THRESHOLD = 0.6

SENDER_EMAIL = ("kamarajengg.edu.in@Gmail.com ")
SENDER_PASSWORD = ("vwvc wsff fbrv umzh ")
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
    .chat-header {
        font-size: 28px;
        color: white;
        text-align: center;
        padding: 10px 0 5px 0;
        font-weight: bold;
    }
    .email-fab {
        position: fixed;
        left: 15px;
        bottom: 20px;
        background-color: #1c1c1c;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        color: white;
        cursor: pointer;
        z-index: 1001;
        box-shadow: 0px 0px 6px #000;
    }
    .email-popup {
        position: fixed;
        left: 80px;
        bottom: 30px;
        background-color: #222;
        padding: 16px;
        border-radius: 10px;
        max-width: 300px;
        z-index: 1002;
        box-shadow: 0 0 5px #000;
    }
    </style>
    <div class="scrolling-banner">
        💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
    </div>
    <div class="chat-header">KCET Assistant</div>
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
    submitted = col2.form_submit_button("\u27a4")

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
        full_response = df.iloc[idx]['Answer']
    else:
        full_response = "❌ Sorry, I couldn't understand that. Please rephrase."

    # Generate TTS audio first
    try:
        tts = gTTS(text=full_response, lang='en')
        audio_file = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        os.remove(audio_file)
    except Exception as e:
        st.error(f"TTS error: {e}")

    # Typing animation simulation
    bot_msg = ""
    placeholder = st.empty()
    for char in full_response:
        bot_msg += char
        placeholder.markdown(f"<div style='background-color:#222; padding:10px; border-radius:10px; text-align:left; margin:5px 0;'>"
                             f"<b>🤖</b>: {bot_msg}</div>", unsafe_allow_html=True)
        time.sleep(0.015)

    st.session_state.chat_log.append(("🤖", full_response))
    st.rerun()

# --- Floating Button to Show Email PDF Form ---
show_email = st.session_state.get("show_email", False)
if st.button("✉️", key="show_email_btn"):
    st.session_state["show_email"] = not show_email
    st.rerun()

if st.session_state.get("show_email"):
    with st.container():
        st.markdown("<div class='email-popup'>", unsafe_allow_html=True)
        email = st.text_input("Email Address", key="email_input")
        if st.button("Send PDF"):
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                pdf = FPDF()
                pdf.add_page()
                pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
                pdf.set_font("DejaVu", size=12)

                def clean_text(text):
                    return ''.join(c if ord(c) < 128 else '?' for c in text)

                for speaker, msg in st.session_state.chat_log:
                    safe_msg = clean_text(f"{speaker}: {msg}")
                    pdf.multi_cell(0, 10, safe_msg)

                filename = f"kcet_chat_{uuid.uuid4().hex}.pdf"
                try:
                    pdf.output(filename)

                    msg = EmailMessage()
                    msg['Subject'] = "KCET Chat Log"
                    msg['From'] = SENDER_EMAIL
                    msg['To'] = email
                    msg.set_content("Here is your chat log with the KCET Assistant.")

                    with open(filename, "rb") as f:
                        msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename="kcet_chat.pdf")

                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                        smtp.send_message(msg)

                    st.success("✅ Email sent successfully!")

                except Exception as e:
                    st.error(f"Email error: {e}")
                finally:
                    if os.path.exists(filename):
                        os.remove(filename)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Floating Action Button UI ---
st.markdown("""
<div class="email-fab">
  ✉️
</div>
""", unsafe_allow_html=True)
