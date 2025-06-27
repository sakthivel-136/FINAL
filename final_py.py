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
def send_email(recipient_emails, subject, body, attachment_path):
    if isinstance(recipient_emails, str):
        recipient_emails = [email.strip() for email in recipient_emails.split(",") if "@" in email]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipient_emails)
    msg.set_content(body)
    with open(attachment_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(attachment_path))
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

@st.cache_data
def load_vector_data():
    if os.path.exists(tf_vector_file):
        with open(tf_vector_file, "rb") as f:
            vectorizer, vectors, df = pickle.load(f)
    else:
        df = pd.read_csv(csv_file)
        df['Question'] = df['Question'].str.lower().str.strip()
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(df['Question'])
        with open(tf_vector_file, "wb") as f:
            pickle.dump((vectorizer, vectors, df), f)
    return vectorizer, vectors, df

vectorizer, vectors, df = load_vector_data()

st.set_page_config(page_title="KCET Chatbot", layout="centered")

if "language" not in st.session_state:
    st.session_state.language = "en"

if "original_log" not in st.session_state:
    st.session_state.original_log = []

if "trigger_rerun" not in st.session_state:
    st.session_state.trigger_rerun = False

if st.session_state.get("trigger_rerun"):
    st.session_state.trigger_rerun = False
    st.experimental_rerun()

# Top bar with logo and title
try:
    with open("kcet_logo.png", "rb") as img:
        logo_base64 = base64.b64encode(img.read()).decode()
        logo_html = f"<img src='data:image/png;base64,{logo_base64}' style='border-radius:50%; width:50px; height:50px; vertical-align:middle; margin-right:10px;'>"
except:
    logo_html = ""

st.markdown(f"""
<div style='display:flex; align-items:center; justify-content:center;'>
    {logo_html}
    <h1 style='display:inline; color:#4CAF50;'>KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY</h1>
</div>
<div style="overflow:hidden; white-space:nowrap; animation:scroll-left 12s linear infinite; background:#333; color:white; padding:8px;">
    💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
</div>
<style>
@keyframes scroll-left {
  0%   { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question...")
    submitted = st.form_submit_button("➔")

if submitted and user_input.strip():
    user_msg = user_input.strip()
    st.session_state.original_log.append(("You", user_msg, "User"))
    vec = vectorizer.transform([user_msg.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()
    answer = df.iloc[idx]['Answer'] if max_sim >= threshold else "❌ Sorry, I couldn't understand that. Please rephrase."
    if st.session_state.language == "ta":
        answer = GoogleTranslator(source='en', target='ta').translate(answer)
    st.session_state.original_log.append(("KCET Assistant", answer, "Assistant"))
    with st.spinner("KCET Assistant typing..."):
        time.sleep(min(1.5, len(answer)/20))
    st.success(answer)
    speak_text(answer)

for speaker, msg, role in st.session_state.original_log:
    align = 'right' if role == "User" else 'left'
    bubble_color = '#d0e8f2' if role == "User" else '#d1d1e9'
    st.markdown(f"""
        <div style='background-color:{bubble_color}; padding:10px; margin:10px; border-radius:10px; text-align:{align};'>
            <b>{speaker}</b>: {msg}
        </div>
    """, unsafe_allow_html=True)

# Export section
with st.expander("📤 Export Chat as TXT/DOC and Email"):
    file_format = st.radio("Select Format", ["TXT", "DOC"])
    recipients = st.text_input("📧 Enter comma-separated emails", key="multi_email")

    def export_chat_to_text_file(ext):
        content = "\n".join([f"{s} ({r}): {m}" for s, m, r in st.session_state.original_log])
        suffix = ".doc" if ext == "DOC" else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as tmp:
            tmp.write(content)
            return tmp.name

    if st.button("Send Email"):
        if recipients and file_format:
            file_path = export_chat_to_text_file(file_format)
            result = send_email(recipients, "KCET Chat Export", f"Attached is your KCET chat export in {file_format} format.", file_path)
            if result is True:
                st.success("✅ Sent successfully to all recipients")
            else:
                st.error(f"❌ Email error: {result}")

# Bottom buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔄 Translate to Tamil" if st.session_state.language == "en" else "🖙 Back to English"):
        st.session_state.language = "ta" if st.session_state.language == "en" else "en"
        st.session_state.trigger_rerun = True

with col2:
    if st.button("🩹 Clear Chat"):
        st.session_state.original_log = []
        st.session_state.trigger_rerun = True
