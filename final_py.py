# KCET Chatbot: Finalized Styling with Pages 1, 2, 3
import streamlit as st
import base64, os, pickle, re, tempfile, time, sqlite3, smtplib
import pandas as pd
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from fpdf import FPDF
from email.message import EmailMessage
import openai

# ========== CONFIG ==========
openai.api_key = "sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl"
ADMIN_PASSWORD = "kcetadmin123"
TFIDF_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
DB_FILE = "kcet_chatlog.db"
THRESHOLD = 0.6
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"

# ========== INITIAL STATE ==========
def init_state():
    defaults = {
        "page": 1, "dark_mode": False, "img_idx": 0, "autoplay_enabled": True, "music_played": False,
        "language": "en", "original_log": [], "last_input": "",
        "username": "You", "user_color": "#d0e8f2", "bot_color": "#d1d1e9",
        "enable_export": True, "admin_page": False, "logged_in": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# ========== DB FUNCTIONS ==========
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chatlog (
            username TEXT, role TEXT, message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT, email TEXT, phone TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ========== UTILS ==========
def save_to_db(user, role, msg):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO chatlog (username, role, message) VALUES (?, ?, ?)", (user, role, msg))
    conn.commit()
    conn.close()

def store_user_info(name, email, phone):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO users (name, email, phone) VALUES (?, ?, ?)", (name, email, phone))
    conn.commit()
    conn.close()
    send_email(email, "Welcome to KCET Chatbot", f"Hi {name},\n\nThanks for logging into the KCET chatbot.\n\nBest,\nKCET Bot Team")

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

# ========== EXPORT ==========
def export_pdf_from_log():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for speaker, msg, role in st.session_state.original_log:
        clean_msg = re.sub(r'[^\x00-\x7F]+', '', msg)
        clean_speaker = re.sub(r'[^\x00-\x7F]+', '', speaker)
        pdf.multi_cell(0, 10, txt=f"{clean_speaker} ({role}): {clean_msg}")
    path = os.path.join(tempfile.gettempdir(), "kcet_chat.pdf")
    pdf.output(path, "F")
    return path

# ========== PAGE 1 ==========
if st.session_state.page == 1:
    st.set_page_config(page_title="KCET Welcome", layout="centered")
    st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")

    logo_path = "kcet_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)

    st.markdown("""
        <div style='text-align:center;'>
            <h2>KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY</h2>
            <p>Welcome to our virtual assistant</p>
        </div>
    """, unsafe_allow_html=True)

    name = st.text_input("Enter your name")
    email = st.text_input("Enter your email")
    phone = st.text_input("Enter your phone")
    if st.button("Start Chat"):
        if name and email and phone:
            store_user_info(name, email, phone)
            st.session_state.username = name
            st.session_state.page = 2
            st.rerun()
        else:
            st.warning("Please fill all fields")

# ========== PAGE 2 ==========
elif st.session_state.page == 2:
    st.set_page_config(page_title="Loading KCET Chatbot", layout="centered")
    st.markdown("""
        <div style='text-align:center; margin-top:100px;'>
            <img src='https://i.gifer.com/VAyR.gif' width='100'>
            <h4>Launching KCET Chatbot...</h4>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.page = 3
    st.rerun()

# ========== PAGE 3 ==========
elif st.session_state.page == 3:
    st.set_page_config(page_title="KCET Chatbot", layout="centered")
    st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")

    bg_color = "#1e1e1e" if st.session_state.dark_mode else "#f4f4f4"
    text_color = "#fff" if st.session_state.dark_mode else "#000"

    st.markdown(f"""
        <div style='text-align:center; background:{bg_color}; color:{text_color}; padding:10px;'>
            <h3>🤖 KCET Chatbot</h3>
        </div>
    """, unsafe_allow_html=True)

    if os.path.exists("kcet_logo.png"):
        st.image("kcet_logo.png", width=80)

    if st.button("🏠 Main Page"):
        st.session_state.page = 1
        st.rerun()

    for speaker, msg, role in st.session_state.original_log:
        align = 'right' if role == "User" else 'left'
        color = st.session_state.user_color if role == "User" else st.session_state.bot_color
        avatar = "🧑‍🎓" if role == "User" else "🤖"
        st.markdown(f"""
            <div style='background-color:{color}; padding:10px; margin:10px; border-radius:10px; text-align:{align};'>
                <b>{avatar} {speaker}</b>: {msg}
            </div>
        """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask your question...", value=st.session_state.last_input)
        submitted = st.form_submit_button("➤")

    if submitted and user_input.strip():
        msg = re.sub(r'[^\x00-\x7F]+', '', user_input.strip())
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
                messages=[{"role": "system", "content": "You are a helpful KCET assistant."},
                         {"role": "user", "content": msg}]
            ).choices[0].message.content

        if st.session_state.language == "ta":
            response = GoogleTranslator(source='en', target='ta').translate(response)

        st.session_state.original_log.append(("KCET Assistant", response, "Assistant"))
        save_to_db("KCET Assistant", "Assistant", response)
        time.sleep(min(2.0, len(response)/30))

        tts = gTTS(text=response, lang='ta' if st.session_state.language == "ta" else 'en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_base64 = base64.b64encode(open(fp.name, "rb").read()).decode()
            audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.original_log.clear()
        st.session_state.last_input = ""
        st.rerun()

    if st.button("🔄 Translate to Tamil" if st.session_state.language == "en" else "↩ Back to English"):
        st.session_state.language = "ta" if st.session_state.language == "en" else "en"
        for i in range(len(st.session_state.original_log)):
            speaker, msg, role = st.session_state.original_log[i]
            if role == "Assistant":
                st.session_state.original_log[i] = (speaker, GoogleTranslator(source='en' if st.session_state.language=='ta' else 'ta', target=st.session_state.language).translate(msg), role)
        st.rerun()

    if st.button("📄 Export Chat as PDF and Email"):
        pdf_path = export_pdf_from_log()
        send_email(st.session_state.username + "@example.com", "KCET Chat Log", "Attached is your chat log.", attachment=pdf_path)
        st.success("PDF exported and emailed successfully!")

    if st.sidebar.button("🔐 Admin Panel"):
        passwd = st.sidebar.text_input("Enter Admin Password", type="password")
        if passwd == ADMIN_PASSWORD:
            st.session_state.page = 4
            st.rerun()

# ========== PAGE 4: ADMIN ==========
elif st.session_state.page == 4:
    st.set_page_config(page_title="Admin Panel", layout="centered")
    st.markdown("<h3>📊 KCET Admin Dashboard</h3>", unsafe_allow_html=True)

    if st.button("⬅ Back to Chat"):
        st.session_state.page = 3
        st.rerun()

    conn = sqlite3.connect(DB_FILE)
    users = pd.read_sql_query("SELECT * FROM users ORDER BY timestamp DESC", conn)
    chats = pd.read_sql_query("SELECT * FROM chatlog ORDER BY timestamp DESC", conn)
    conn.close()

    st.subheader("📋 Registered Users")
    st.dataframe(users)

    st.subheader("💬 Chat Logs")
    st.dataframe(chats)

    if st.button("⬇ Download All as CSV"):
        users.to_csv("users.csv", index=False)
        chats.to_csv("chatlog.csv", index=False)
        st.success("Files saved as users.csv and chatlog.csv")
