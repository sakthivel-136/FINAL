import streamlit as st
import pandas as pd
import pickle
import os
import smtplib
import re
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from gtts import gTTS
import tempfile
import base64
from deep_translator import GoogleTranslator
import time

# ========== EMAIL CREDENTIALS ==========
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"  # Gmail App Password

# ========== Remove Emojis ==========
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# ========== Voice Response ==========
def speak_text(text):
    lang = 'ta' if st.session_state.get("language", "en") == "ta" else 'en'
    tts = gTTS(text=text, lang=lang, tld='com')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_base64 = base64.b64encode(open(fp.name, "rb").read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# ========== Email Sending ==========
def send_email(recipient_email, subject, body, attachment_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg.set_content(body)
    with open(attachment_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="KCET_Chat_Report.pdf")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        return str(e)

# ========== Constants ==========
tf_vector_file = "vectorized.pkl"
csv_file = "kcet.csv"
threshold = 0.6

# ========== Translation Support ==========
def translate_text(text, lang):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text

def translate_chat_log(target_lang):
    if "original_log" not in st.session_state:
        return
    translated_log = []
    for i, (speaker, msg, role) in enumerate(st.session_state.original_log):
        if role == "Assistant":
            msg = translate_text(msg, target_lang)
        translated_log.append((speaker, msg, role))
    st.session_state.chat_log = translated_log

# ========== Page Setup ==========
st.set_page_config(page_title="KCET Chatbot", layout="centered")

# ========== Session Defaults ==========
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "language" not in st.session_state:
    st.session_state.language = "en"
if "original_log" not in st.session_state:
    st.session_state.original_log = []
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]

# ========== Theme and Language Toggle ==========
st.sidebar.toggle("🍗 Theme", value=st.session_state.theme == "Dark", key="theme_toggle")
st.session_state.theme = "Dark" if st.session_state.theme_toggle else "Light"

if st.sidebar.button("🔄 Translate to " + ("English" if st.session_state.language == "ta" else "Tamil")):
    new_lang = "en" if st.session_state.language == "ta" else "ta"
    translate_chat_log(new_lang)
    st.session_state.language = new_lang

# ========== Theme Settings ==========
mode = st.session_state.theme
is_dark = mode == "Dark"
bg_color = "#111" if is_dark else "#fff"
final_txt_color = "white" if is_dark else "black"

# ========== Sidebar ==========
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.user_name = st.text_input("👤 Your Name", value=st.session_state.get("user_name", "Shakthivel"))
    user_bubble_color = st.color_picker("🎨 User Bubble", "#d0e8f2")
    assistant_bubble_color = st.color_picker("🎨 Assistant Bubble", "#d1d1e9")
    text_color = st.color_picker("🕋️ Text Color", "#000000")

# ========== Export Toggle Button ==========
st.markdown("""
    <style>
    .circle-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 12px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 50%;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .circle-button:hover {
        transform: scale(1.2);
    }
    .message {
        padding: 20px;
        border-radius: 12px;
        margin: 6px 0;
        animation: fadein 0.5s;
    }
    @keyframes fadein {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .scrolling-banner {
        overflow: hidden;
        white-space: nowrap;
        animation: scroll-left 20s linear infinite;
        color: gold;
        background-color: """ + bg_color + """;
        padding: 8px;
        font-weight: bold;
        font-size: 16px;
        text-align: center;
        direction: """ + ("rtl" if st.session_state.language == "ta" else "ltr") + """;
    }
    @keyframes scroll-left {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    <button class="circle-button" onclick="document.getElementById('export-section').style.display='block'">📄</button>
    <div class="scrolling-banner">
        💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
    </div>
    <script>
    setTimeout(function(){ window.scrollTo(0, document.body.scrollHeight); }, 500);
    </script>
""", unsafe_allow_html=True)

# ========== Header UI ==========
st.markdown(f"""
<div class="chat-header" style="text-align:center; font-size:30px; color:{final_txt_color}; font-weight:bold; margin-top: 10px;">
    KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY - KCET ChatBot
</div>
""", unsafe_allow_html=True)

# ========== AI Typing Animation ==========
if "typing" in st.session_state and st.session_state.typing:
    with st.spinner("KCET Assistant is typing..."):
        time.sleep(1.5)
    st.session_state.typing = False

# ========== Export Section: Tamil-English Side-by-Side PDF ==========
if st.button("📄 Export Chat as PDF with Tamil-English"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="KCET Chat Transcript", ln=True, align='C')
    pdf.ln(5)
    for speaker, msg, role in st.session_state.original_log:
        en_msg = msg
        ta_msg = translate_text(msg, "ta")
        pdf.set_font("Arial", style='B', size=11)
        pdf.multi_cell(0, 8, f"{speaker} ({role}):", align='L')
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 8, f"EN: {en_msg}")
        pdf.multi_cell(0, 8, f"TA: {ta_msg}")
        pdf.ln(4)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            st.download_button("📅 Download Tamil-English PDF", f.read(), file_name="KCET_Chat_Transcript.pdf", mime="application/pdf")
