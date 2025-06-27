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

# ========== EMAIL CREDENTIALS ==========
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"  # Gmail App Password

# ========== Remove Emojis ==========
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# ========== Voice Response ==========
def speak_text(text):
    tts = gTTS(text=text, lang='en')
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

st.set_page_config(page_title="KCET Chatbot", layout="centered")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ========== Theme Switch ==========
st.markdown("""
    <style>
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0;
            right: 0; bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #2196F3;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
    </style>
    <label class="switch">
      <input type="checkbox" onchange="window.location.reload();" {}>
      <span class="slider"></span>
    </label>
""".format("checked" if st.session_state.theme == "Dark" else ""), unsafe_allow_html=True)

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
    </style>
    <button class="circle-button" onclick="document.getElementById('export-section').style.display='block'">📤</button>
""", unsafe_allow_html=True)

# ========== Header UI ==========
st.markdown(f"""
<div class="scrolling-banner" style="background-color:{bg_color}; color:gold; padding:10px; text-align:center;">
    💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
</div>
<div class="chat-header" style="text-align:center; font-size:24px; color:{final_txt_color}; font-weight:bold;">KCET ChatBot</div>
""", unsafe_allow_html=True)

# ========== Load Vectorized Data ==========
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
    st.session_state.chat_log.append(("KCET Assistant", response, "Assistant"))
    speak_text(response)
    st.experimental_rerun()

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
    st.experimental_rerun()
