# KCET Chatbot Full App: Pages 1, 2, 3, 4
import streamlit as st
import os, base64, re, time, pickle, tempfile, smtplib, sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from deep_translator import GoogleTranslator
from fpdf import FPDF
from email.message import EmailMessage
import openai

# ========== CONFIGURATION ==========
openai.api_key = "sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl"
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
        "admin_triggered": False
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

# ========== PAGE 1 ==========
if st.session_state.page == 1:
    st.set_page_config("KCET Welcome", layout="centered")
    if os.path.exists("kcet_logo.png"):
        st.image("kcet_logo.png", width=120)
    st.title("KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY")
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Chat"):
            if name and email and phone:
                store_user_info(name, email, phone)
                st.session_state.username = name
                st.session_state.page = 2
                st.rerun()
            else:
                st.warning("Fill all fields")
    with col2:
        if st.button("Admin Panel"):
            st.session_state.admin_triggered = True

    if st.session_state.admin_triggered:
        st.session_state.admin_pass = st.text_input("Enter Admin Password", type="password")
        if st.session_state.admin_pass:
            if st.session_state.admin_pass == ADMIN_PASSWORD:
                st.session_state.page = 4
                st.rerun()
            else:
                st.error("Incorrect password")

# ========== PAGE 2 ==========
elif st.session_state.page == 2:
    st.title("Chat Loading")
    st.session_state.export_email = st.text_input("Enter email to receive chat PDF")
    if st.button("Continue to Chat"):
        if st.session_state.export_email:
            st.session_state.page = 3
            st.rerun()
        else:
            st.warning("Please enter an email")

# ========== PAGE 3 ==========
elif st.session_state.page == 3:
    st.title("KCET Chatbot 🤖")
    if st.button("🏠 Home"): st.session_state.page = 1; st.rerun()

    # Theme picker only in Page 3
    st.sidebar.header("🎨 Bubble Theme")
    st.session_state.user_color = st.sidebar.color_picker("User Bubble", st.session_state.user_color)
    st.session_state.bot_color = st.sidebar.color_picker("Bot Bubble", st.session_state.bot_color)

    for speaker, msg, role in st.session_state.original_log:
        align = 'right' if role == "User" else 'left'
        bg_color = st.session_state.user_color if role == "User" else st.session_state.bot_color
        avatar = "🧑‍🎓" if role == "User" else "🤖"
        st.markdown(f"<div style='text-align:{align}; background:{bg_color}; padding:10px; border-radius:10px; margin:5px;'>" +
                    f"<b>{avatar} {speaker}</b>: {msg}</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask your question...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        msg = remove_emojis(user_input.strip())
        st.session_state.original_log.append((st.session_state.username, msg, "User"))
        save_to_db(st.session_state.username, "User", msg)

        if os.path.exists(TFIDF_FILE):
            with open(TFIDF_FILE, "rb") as f:
                vectorizer, vectors, df = pickle.load(f)
        else:
            df = pd.read_csv(CSV_FILE)
            df['Question'] = df['Question'].str.lower().str.strip()
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(df['Question'])
            with open(TFIDF_FILE, "wb") as f:
                pickle.dump((vectorizer, vectors, df), f)

        vec = vectorizer.transform([msg.lower()])
        sim = cosine_similarity(vec, vectors)
        idx = sim.argmax()
        max_sim = sim.max()

        if max_sim >= THRESHOLD:
            response = df.iloc[idx]['Answer']
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": msg}]
            ).choices[0].message.content

        if st.session_state.language == "ta":
            response = GoogleTranslator(source='en', target='ta').translate(response)

        st.session_state.original_log.append(("KCET Assistant", response, "Assistant"))
        save_to_db("KCET Assistant", "Assistant", response)

        tts = gTTS(text=response, lang='ta' if st.session_state.language == "ta" else 'en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_base64 = base64.b64encode(open(fp.name, "rb").read()).decode()
            st.markdown(f"<audio autoplay><source src='data:audio/mp3;base64,{audio_base64}' type='audio/mp3'></audio>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Download PDF"):
            pdf_path = export_pdf_from_log()
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="kcet_chat.pdf")
    with col2:
        if st.button("📧 Email PDF"):
            pdf_path = export_pdf_from_log()
            send_email(st.session_state.export_email, "KCET Chat Log", "Here is your chat log.", pdf_path)
            st.success("Email sent")

# ========== PAGE 4: Admin Dashboard ==========
elif st.session_state.page == 4:
    st.set_page_config(page_title="KCET Admin Dashboard", layout="wide")
    st.markdown("""
        <div style='text-align:center; padding: 10px;'>
            <h2>🛠️ KCET Admin Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)

    if os.path.exists("kcet_logo.png"):
        st.image("kcet_logo.png", width=100)

    if st.button("⬅ Back to Welcome"):
        st.session_state.page = 1
        st.rerun()

    tab1, tab2 = st.tabs(["📊 User Details", "💬 Chat Logs"])

    with tab1:
        st.subheader("👤 Registered Users")
        conn = sqlite3.connect(DB_FILE)
        users_df = pd.read_sql_query("SELECT * FROM users", conn)
        st.dataframe(users_df)

        col1, col2 = st.columns(2)
        with col1:
            csv = users_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Users CSV", csv, "users.csv", "text/csv")
        with col2:
            if st.button("📧 Email Users CSV"):
                tmp_file = os.path.join(tempfile.gettempdir(), "users.csv")
                users_df.to_csv(tmp_file, index=False)
                send_email(SENDER_EMAIL, "KCET Users List", "Attached is the user list.", tmp_file)
                st.success("✅ User list emailed!")

    with tab2:
        st.subheader("📜 All Chat Logs")
        logs_df = pd.read_sql_query("SELECT * FROM chatlog ORDER BY timestamp DESC", conn)
        st.dataframe(logs_df)

        col1, col2 = st.columns(2)
        with col1:
            log_csv = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Logs CSV", log_csv, "chat_logs.csv", "text/csv")
        with col2:
            if st.button("📧 Email Logs CSV"):
                tmp_log = os.path.join(tempfile.gettempdir(), "chat_logs.csv")
                logs_df.to_csv(tmp_log, index=False)
                send_email(SENDER_EMAIL, "KCET Chat Logs", "Attached are the full chat logs.", tmp_log)
                st.success("✅ Chat logs emailed!")

        conn.close()
