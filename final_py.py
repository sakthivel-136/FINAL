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

# ========== PAGE ROUTING LOGIC ==========
if st.session_state.page == 1:
    st.set_page_config(page_title="KCET Welcome", layout="centered")
    st.image("kcet_logo.png", width=120)
    st.title("KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY")
    play_welcome_audio()
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<style>.stButton > button {background-color:#0052cc;color:white;border-radius:8px;padding:10px 16px;font-weight:bold;}</style>""", unsafe_allow_html=True)
        if st.button("🎓 Enter Chatbot"):
            st.session_state.page = 2
            st.rerun()
    with col2:
        if st.button("🔐 Admin Panel"):
            st.session_state.admin_triggered = True
            st.session_state.page = 4
            st.rerun()

elif st.session_state.page == 2:
    st.title("📩 Enter Your Email to Receive Export")
    st.session_state.export_email = st.text_input("Email Address", value=st.session_state.export_email)
    if st.button("➡ Go to Chatbot"):
        st.session_state.page = 3
        st.rerun()

elif st.session_state.page == 3:
    st.set_page_config(page_title="KCET Chatbot", layout="wide")
    st.title("🤖 KCET Chatbot")
    with st.sidebar:
        st.text_input("Your Name", key="username")
        st.color_picker("User Bubble Color", key="user_color")
        st.color_picker("Bot Bubble Color", key="bot_color")
        st.radio("Language", options=["en", "ta"], key="language", horizontal=True)
        if st.button("🗑 Clear Chat"):
            st.session_state.original_log.clear()

    for speaker, msg, role in st.session_state.original_log:
        color = st.session_state.user_color if role == "User" else st.session_state.bot_color
        align = "right" if role == "User" else "left"
        st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:10px; margin:10px; text-align:{align};'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask your question:", key="last_input")
        submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            msg = user_input.strip()
            if st.session_state.language == "ta":
                msg_translated = GoogleTranslator(source='ta', target='en').translate(msg)
            else:
                msg_translated = msg

            df = pd.read_csv(CSV_FILE)
            df['Question'] = df['Question'].str.lower()
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(df['Question'])
            input_vec = vectorizer.transform([msg_translated.lower()])
            scores = cosine_similarity(input_vec, vectors).flatten()
            best_idx = scores.argmax()
            if scores[best_idx] > THRESHOLD:
                reply = df.iloc[best_idx]['Answer']
            else:
                reply = "Sorry, I couldn't understand that."

            if st.session_state.language == "ta":
                reply = GoogleTranslator(source='en', target='ta').translate(reply)

            st.session_state.original_log.append((st.session_state.username, msg, "User"))
            st.session_state.original_log.append(("KCET Assistant", reply, "Assistant"))
            save_to_db(st.session_state.username, "User", msg)
            save_to_db("KCET Assistant", "Assistant", reply)
            st.rerun()

    st.markdown("""<style>.stButton > button {background-color:#28a745;color:white;border-radius:6px;padding:8px 16px;margin:5px;}</style>""", unsafe_allow_html=True)
    if st.button("📄 Export PDF"):
        pdf_path = export_pdf_from_log()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Chat PDF", f, file_name="kcet_chat.pdf")

    if st.button("📤 Email PDF"):
        pdf_path = export_pdf_from_log()
        send_email(st.session_state.export_email, "KCET Chat Transcript", "Your chat export is attached.", pdf_path)
        st.success("✅ PDF exported and emailed successfully!")

elif st.session_state.page == 4:
    st.title("🔐 Admin Panel")
    if not st.session_state.get("admin_authenticated"):
        st.session_state.admin_pass = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if st.session_state.admin_pass == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.success("✅ Welcome Admin")
        with sqlite3.connect(DB_FILE) as conn:
            users_df = pd.read_sql("SELECT * FROM users", conn)
            chats_df = pd.read_sql("SELECT * FROM chatlog", conn)
        st.subheader("📘 User Info")
        st.dataframe(users_df)
        st.subheader("💬 Chat Logs")

        filter_user = st.selectbox("Filter by User", options=["All"] + list(users_df['name'].unique()))
        filtered_chats = chats_df if filter_user == "All" else chats_df[chats_df['username'] == filter_user]
        st.dataframe(filtered_chats)

        st.subheader("📊 Chat Activity")
        chat_counts = chats_df['username'].value_counts()
        fig, ax = plt.subplots()
        chat_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Messages per User")
        ax.set_ylabel("Message Count")
        st.pyplot(fig)

        if st.button("📤 Send All Logs to Admin Email"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                chats_df.to_csv(tmp.name, index=False)
                send_email(SENDER_EMAIL, "All Chat Logs", "Attached all chat logs from the chatbot.", tmp.name)
                st.success("✅ Sent all logs to Admin Email")
