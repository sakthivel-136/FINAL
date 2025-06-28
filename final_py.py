# KCET Chatbot Full App: Pages 1, 2, 3, 4 with Analytics and Audio
import streamlit as st
import os, base64, re, time, pickle, tempfile, smtplib, sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from deep_translator import GoogleTranslator
from fpdf import FPDF
from email.message import EmailMessage
import matplotlib.pyplot as plt
from PIL import Image

# ========== CONFIGURATION ==========
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"
ADMIN_PASSWORD = "qwerty12345"
DB_FILE = "kcet_chatlog.db"
CSV_FILE = "kcet.csv"
TFIDF_FILE = "vectorized.pkl"
THRESHOLD = 0.6

# ========== INITIALIZATION ==========
def init_state():
    defaults = {
        "page": 1, "img_idx": 0, "username": "You", "language": "en",
        "original_log": [], "last_input": "", "user_color": "#d0e8f2",
        "bot_color": "#d1d1e9", "export_email": "", "admin_pass": "",
        "admin_triggered": False, "admin_authenticated": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chatlog (
            username TEXT, role TEXT, message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT, email TEXT, phone TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.commit()
    conn.close()

init_state()
init_db()

# ========== UTILITY FUNCTIONS ==========
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def store_user_info(name, email, phone):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO users (name, email, phone) VALUES (?, ?, ?)", (name, email, phone))
    conn.commit()
    conn.close()
    send_email(email, "Welcome to KCET Chatbot", f"Hi {name},\nThanks for logging in.")

def send_email(to_email, subject, body, attachment=None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg.set_content(body)
    if attachment and os.path.exists(attachment):
        with open(attachment, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(attachment))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.warning(f"Email error: {e}")

def save_to_db(user, role, msg):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO chatlog (username, role, message) VALUES (?, ?, ?)", (user, role, msg))
    conn.commit()
    conn.close()

def export_pdf_from_log():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for speaker, msg, role in st.session_state.original_log:
        pdf.multi_cell(0, 10, txt=f"{remove_emojis(speaker)} ({role}): {remove_emojis(msg)}")
    path = os.path.join(tempfile.gettempdir(), "kcet_chat.pdf")
    pdf.output(path)
    return path

def play_welcome_audio():
    if os.path.exists("kcet_music.mp3"):
        audio_bytes = open("kcet_music.mp3", "rb").read()
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>", unsafe_allow_html=True)

# ========== PAGE 1 ==========
if st.session_state.page == 1:
    st.set_page_config(page_title="KCET Welcome", layout="centered")
    st.title("🎓 Welcome to KCET Chatbot")

    st.image("kcet_logo.png", width=120)
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    phone = st.text_input("Your Phone Number")

    if st.button("🚀 Enter Chatbot", use_container_width=True):
        if name and email:
            st.session_state.username = name
            store_user_info(name, email, phone)
            st.session_state.page = 2
            st.rerun()
        else:
            st.warning("Please fill all required details")

    if st.button("🔐 Admin Panel", use_container_width=True):
        st.session_state.admin_triggered = True

    if st.session_state.admin_triggered:
        st.session_state.admin_pass = st.text_input("Enter Admin Password", type="password")
        if st.button("Submit"):
            if st.session_state.admin_pass == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.session_state.page = 4
                st.rerun()
            else:
                st.error("Incorrect password")

# ========== PAGE 2 ==========
if st.session_state.page == 2:
    st.title("🤖 KCET Assistant")
    play_welcome_audio()

    email_to = st.text_input("📧 Enter your email to receive export")
    if email_to:
        st.session_state.export_email = email_to

    if st.button("➡️ Start Chat", use_container_width=True):
        st.session_state.page = 3
        st.rerun()

# ========== PAGE 3 ==========
if st.session_state.page == 3:
    st.title("💬 Chat with KCET Bot - Page 3")

    with st.sidebar:
        theme_label = st.radio("🎨 Select Chat Theme", ["Modern Blue", "Mint Green", "Elegant Gold"])

    themes = {
        "Modern Blue": ("#e1ecf4", "#d4d9f5"),
        "Mint Green": ("#d2f4e3", "#e8f6f0"),
        "Elegant Gold": ("#fff4cc", "#e0f7fa")
    }

    user_color, bot_color = themes[theme_label]
    st.session_state.user_color = user_color
    st.session_state.bot_color = bot_color
    st.caption(f"Theme: {theme_label}")

    user_input = st.text_input("Type your message")
    if user_input:
        vec_data = pd.read_csv(CSV_FILE)
        questions = vec_data['Question'].str.lower().str.strip()
        answers = vec_data['Answer']
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(questions)
        input_vec = vectorizer.transform([user_input.lower()])
        similarity = cosine_similarity(input_vec, vectors)
        idx = similarity.argmax()
        max_sim = similarity.max()
        answer = answers[idx] if max_sim >= THRESHOLD else "Sorry, I couldn't understand that."

        if st.session_state.language == "ta":
            answer = GoogleTranslator(source='en', target='ta').translate(answer)

        st.session_state.original_log.append((st.session_state.username, user_input, "User"))
        st.session_state.original_log.append(("KCET Bot", answer, "Assistant"))
        save_to_db(st.session_state.username, "User", user_input)
        save_to_db("KCET Bot", "Assistant", answer)
        st.rerun()

    for speaker, msg, role in st.session_state.original_log:
        align = 'right' if role == "User" else 'left'
        color = st.session_state.user_color if role == "User" else st.session_state.bot_color
        st.markdown(f"""
            <div style='background-color:{color}; padding:10px; margin:10px; border-radius:10px; text-align:{align};'>
                <b>{speaker}</b>: {msg}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Export Chat to PDF", use_container_width=True):
            pdf_path = export_pdf_from_log()
            st.success("PDF exported!")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="kcet_chat.pdf")
    with col2:
        if st.button("📧 Send Email", use_container_width=True):
            if st.session_state.export_email:
                pdf_path = export_pdf_from_log()
                send_email(st.session_state.export_email, "KCET Chat Log", "Attached is your chat.", pdf_path)
                st.success("Email sent!")
            else:
                st.warning("Enter your email above")

    if st.button("🏠 Back to Main Page", use_container_width=True):
        st.session_state.page = 1
        st.rerun()

# ========== PAGE 4 ==========
if st.session_state.page == 4:
    st.title("🛠️ Admin Dashboard")

    conn = sqlite3.connect(DB_FILE)
    users = pd.read_sql_query("SELECT * FROM users", conn)
    logs = pd.read_sql_query("SELECT * FROM chatlog", conn)
    conn.close()

    st.subheader("Registered Users")
    st.dataframe(users)

    st.subheader("Chat Logs")
    st.dataframe(logs)

    st.subheader("Chat Analytics")
    count_by_user = logs['username'].value_counts()
    st.bar_chart(count_by_user)

    if st.button("📧 Email All Logs", use_container_width=True):
        csv_path = os.path.join(tempfile.gettempdir(), "kcet_logs.csv")
        logs.to_csv(csv_path, index=False)
        send_email(SENDER_EMAIL, "KCET Logs", "Attached are all chat logs.", csv_path)
        st.success("Logs emailed to admin")

    if st.button("🏠 Back to Main Page", use_container_width=True):
        st.session_state.page = 1
        st.rerun()
