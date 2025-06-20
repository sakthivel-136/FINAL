import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import time
import base64
import smtplib
import math
from datetime import datetime
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from gtts import gTTS

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

VECTOR_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
THRESHOLD = 0.6
SENDER_EMAIL = ("kamarajengg.edu.in@Gmail.com ")
SENDER_PASSWORD = ("vwvc wsff fbrv umzh ")

st.set_page_config(page_title="KCET Chatbot", layout="centered")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2922/2922506.png", width=100)
    st.title("⚙️ Settings")
    mode = st.radio("Select Theme", ["Dark", "Light"], index=0)
    export_option = st.checkbox("Enable Export Options")

dark_mode = mode == "Dark"
background = "#111" if dark_mode else "#fff"
text_color = "white" if dark_mode else "black"
user_bg = "#444" if dark_mode else "#ccc"
bot_bg = "#222" if dark_mode else "#eee"

st.markdown(f"""
    <style>
    .scrolling-banner {{
        overflow: hidden;
        white-space: nowrap;
        animation: scroll-left 20s linear infinite;
        color: gold;
        background-color: {background};
        padding: 8px;
        font-weight: bold;
        font-size: 16px;
        text-align: center;
    }}
    @keyframes scroll-left {{
        0% {{ transform: translateX(100%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    .chat-header {{
        font-size: 28px;
        color: {text_color};
        text-align: center;
        padding: 10px 0 5px 0;
        font-weight: bold;
    }}
    .message {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        animation: fadein 0.5s;
    }}
    @keyframes fadein {{
        from {{opacity: 0; transform: translateY(10px);}}
        to {{opacity: 1; transform: translateY(0);}}
    }}
    .avatar {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
    }}
    </style>
    <div class="scrolling-banner">
        💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
    </div>
    <div class="chat-header">KCET Assistant</div>
""", unsafe_allow_html=True)

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

if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("🧠", "Hello! I'm your KCET Assistant. Ask me anything.")]

# --- Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your question here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Display Chat ---
st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
for speaker, msg in st.session_state.chat_log:
    align = 'right' if speaker == '👤' else 'left'
    bg = user_bg if speaker == '👤' else bot_bg
    avatar = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png" if speaker == "👤" else "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
    st.markdown(f"""
    <div class='message' style='background-color:{bg}; text-align:{align}; color:{text_color};'>
        <img src='{avatar}' class='avatar'/>
        <div><b>{speaker}</b>: {msg}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if submitted and user_input.strip():
    st.session_state.chat_log.append(("👤", user_input.strip()))

    vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    full_response = df.iloc[idx]['Answer'] if max_sim >= THRESHOLD else "❌ Sorry, I couldn't understand that. Please rephrase."

    try:
        tts = gTTS(text=full_response, lang='en')
        audio_file = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)

        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0)

        if AudioSegment:
            duration = AudioSegment.from_file(audio_file).duration_seconds
            time.sleep(math.ceil(duration))

        os.remove(audio_file)
    except Exception as e:
        st.error(f"TTS error: {e}")

    st.session_state.chat_log.append(("🧠", full_response))
    st.rerun()

if export_option:
    st.subheader("📤 Export Chat")
    email = st.text_input("Email Address")
    file_type = st.selectbox("Choose file format", ["PDF", "TXT", "DOC"])
    custom_name = st.text_input("Enter custom file name (without extension)")
    if st.button("Send"):
        if not email or "@" not in email:
            st.error("Please enter a valid email address.")
        else:
            filename = custom_name or f"kcet_chat_{uuid.uuid4().hex}"
            attachment_path = ""
            try:
                if file_type == "PDF":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
                    pdf.set_font("DejaVu", size=12)
                    for speaker, msg in st.session_state.chat_log:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        pdf.multi_cell(0, 10, f"[{timestamp}] {speaker}: {msg}")
                    attachment_path = f"{filename}.pdf"
                    pdf.output(attachment_path)

                elif file_type == "TXT":
                    attachment_path = f"{filename}.txt"
                    with open(attachment_path, "w", encoding="utf-8") as f:
                        for speaker, msg in st.session_state.chat_log:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            f.write(f"[{timestamp}] {speaker}: {msg}\n")

                elif file_type == "DOC":
                    attachment_path = f"{filename}.doc"
                    with open(attachment_path, "w", encoding="utf-8") as f:
                        for speaker, msg in st.session_state.chat_log:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            f.write(f"[{timestamp}] {speaker}: {msg}\n")

                msg = EmailMessage()
                msg['Subject'] = "KCET Chat Log"
                msg['From'] = SENDER_EMAIL
                msg['To'] = email
                msg.set_content("Here is your chat log with the KCET Assistant.")

                with open(attachment_path, "rb") as f:
                    maintype, subtype = ("application", "octet-stream")
                    if file_type == "PDF":
                        subtype = "pdf"
                    elif file_type == "TXT":
                        subtype = "plain"
                    elif file_type == "DOC":
                        subtype = "msword"
                    msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(attachment_path))

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                    smtp.send_message(msg)

                st.success("✅ Email sent successfully!")

            except Exception as e:
                st.error(f"Email error: {e}")
            finally:
                if os.path.exists(attachment_path):
                    os.remove(attachment_path)

if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("🧠", "Hello! I'm your KCET Assistant. Ask me anything.")]
    st.rerun()
