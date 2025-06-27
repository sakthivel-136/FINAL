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

# ========== EMAIL CREDENTIALS ==========
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"  # Gmail App Password

# ========== Remove Emojis ==========
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# ========== Voice Response ==========
def speak_text(text):
    tts = gTTS(text=text, lang='ta' if st.session_state.get("language", "en") == "ta" else 'en')
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

def translate_text(text, lang):
    return GoogleTranslator(source='auto', target=lang).translate(text)

def translate_chat_log(target_lang):
    translated_log = []
    for speaker, msg, role in st.session_state.chat_log:
        if role == "Assistant":
            msg = translate_text(msg, target_lang)
        translated_log.append((speaker, msg, role))
    st.session_state.chat_log = translated_log

st.set_page_config(page_title="KCET Chatbot", layout="centered")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "language" not in st.session_state:
    st.session_state.language = "en"

# ========== Theme and Language Toggle ==========
st.sidebar.toggle("🌗 Theme", value=st.session_state.theme == "Dark", key="theme_toggle")
st.session_state.theme = "Dark" if st.session_state.theme_toggle else "Light"

if st.sidebar.button("🔄 Translate to " + ("English" if st.session_state.language == "ta" else "Tamil")):
    new_lang = "en" if st.session_state.language == "ta" else "ta"
    translate_chat_log(new_lang)
    st.session_state.language = new_lang

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
    text_color = st.color_picker("🖋️ Text Color", "#000000")

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
    }
    .message {
        padding: 14px;
        border-radius: 10px;
        margin: 5px 0;
        animation: fadein 0.5s;
    }
    @keyframes fadein {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    <button class="circle-button" onclick="document.getElementById('export-section').style.display='block'">📤</button>
    <script>
        window.scrollTo(0, document.body.scrollHeight);
    </script>
""", unsafe_allow_html=True)

# ========== Header UI ==========
st.markdown(f"""
<div class="scrolling-banner" style="background-color:{bg_color}; color:gold; padding:10px; text-align:center;">
    💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
</div>
<div class="chat-header" style="text-align:center; font-size:28px; color:{final_txt_color}; font-weight:bold; margin-top: 10px;">KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY - KCET ChatBot</div>
""", unsafe_allow_html=True)

# ========== Load Vectorized Data ==========@st.cache_data
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

if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]

# ========== Chat Form ==========
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Ask your question...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

if submitted and user_input.strip():
    st.session_state.chat_log.append((st.session_state.user_name, user_input.strip(), "User"))
    vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()
    response = df.iloc[idx]['Answer'] if max_sim >= threshold else "❌ Sorry, I couldn't understand that. Please rephrase."
    if st.session_state.language == "ta":
        response = translate_text(response, "ta")
    st.session_state.chat_log.append(("KCET Assistant", response, "Assistant"))
    speak_text(response)
    st.rerun()

# ========== Display Chat ==========
st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
for speaker, msg, role in st.session_state.chat_log:
    align = 'right' if role == "User" else 'left'
    bg = user_bubble_color if role == "User" else assistant_bubble_color
    st.markdown(f"""
    <div class='message' style='background-color:{bg}; text-align:{align}; color:{final_txt_color};'>
        <div><b>{speaker}</b> ({role}): {msg.replace('\xa0', ' ')}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ========== Export Option ==========
st.markdown("<div id='export-section' style='display:none;'>", unsafe_allow_html=True)
st.subheader("📤 Export Chat")
file_type = st.radio("File Type", ["PDF", "TXT", "DOC"], index=0)
email = st.text_input("📧 Email (PDF only)", placeholder="example@gmail.com")

if st.button("Download / Email"):
    try:
        filename = f"{st.session_state.user_name}_chatlog.{file_type.lower()}"
        file_path = None

        if file_type == "PDF":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", '', 12)
            for speaker, msg, role in st.session_state.chat_log:
                clean_msg = remove_emojis(msg.replace('\xa0', ' '))
                pdf.multi_cell(0, 10, f"{speaker} ({role}): {clean_msg}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.output(tmp.name)
                file_path = tmp.name
                with open(file_path, "rb") as f:
                    st.download_button("📥 Download", f.read(), file_name=filename, mime="application/pdf")

            if email and "@" in email:
                result = send_email(email, f"{st.session_state.user_name} Chat Log", "Attached is your KCET chat report.", file_path)
                if result is True:
                    st.success("✅ Email sent to " + email)
                else:
                    st.error(f"❌ Email failed: {result}")

        else:
            text = "\n".join([f"{s} ({r}): {m}" for s, m, r in st.session_state.chat_log])
            mime = "application/msword" if file_type == "DOC" else "text/plain"
            st.download_button("📥 Download", text.encode(), file_name=filename, mime=mime)

    except Exception as e:
        st.error(f"❌ Export failed: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# ========== Clear Chat ==========
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]
    st.rerun()
