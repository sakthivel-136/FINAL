# KCET Chatbot Full App - Setup & Page 1, 2, 3, 4

import streamlit as st
import os, base64, re, time, pickle, tempfile, smtplib, sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from deep_translator import GoogleTranslator
from fpdf import FPDF
from email.message import EmailMessage

# ========== CONFIG ==========
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"
ADMIN_PASSWORD = "qwerty12345"
DB_FILE = "kcet_chatlog.db"
CSV_FILE = "kcet.csv"
THRESHOLD = 0.6

# ========== INIT ==========
def init_state():
    defaults = {
        "page": 1, "username": "You", "language": "en", "original_log": [],
        "export_email": "", "admin_pass": "", "admin_triggered": False,
        "admin_authenticated": False
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

def transition_effect():
    st.markdown("""
        <style>
        div.stButton > button {
            transition: all 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)

def show_custom_loader():
    st.markdown("""
        <style>
        .loader-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(255,255,255,0.95);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .loader {
            border: 12px solid #f3f3f3;
            border-top: 12px solid #3498db;
            border-radius: 50%;
            width: 100px;
            height: 100px;
            animation: spin 1.2s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        <div class="loader-wrapper" id="loader">
            <div class="loader"></div>
        </div>
        <script>
        setTimeout(function() {
            var loader = document.getElementById("loader");
            if (loader) {
                loader.style.display = "none";
            }
        }, 1500);
        </script>
    """, unsafe_allow_html=True)

def play_welcome_audio():
    pass  # You can define your audio logic here

init_state()
init_db()

# ========== PAGE 1 ==========
if st.session_state.page == 1:
    transition_effect()
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("kcet_logo.png", width=60)
    with col2:
        st.title(" Welcome to KCET Chatbot")

    st.markdown("""
        <style>
            input, button { border-radius: 10px !important; font-size: 16px !important; }
        </style>
    """, unsafe_allow_html=True)

    name = st.text_input("🧑 Your Name")
    email = st.text_input("📧 Your Email")
    phone = st.text_input("📱 Phone Number")

    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("🚀 Enter Chatbot", use_container_width=True):
            if name and email:
                st.session_state.username = name
                st.session_state.user_email = email
                st.session_state.user_phone = phone
                store_user_info(name, email, phone)
                send_email(email, "KCET Chatbot Confirmation", f"Hi {name}, This is a CONFIRMATION mail regarding your Login in KCET Chatbot!!\n\nDetails:\nName: {name}\nEmail: {email}\nPhone: {phone}\n\nThanks for Connecting with us!")
                st.session_state.page = 2
                st.rerun()
            else:
                st.warning("Please fill all fields to continue.")

    with col4:
        if st.button("🔐 Admin Panel", use_container_width=True):
            st.session_state.admin_triggered = True

    if st.session_state.admin_triggered:
        st.session_state.admin_pass = st.text_input("Enter Admin Password", type="password")
        if st.button("✅ Submit", use_container_width=True):
            if st.session_state.admin_pass == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.session_state.page = 4
                st.rerun()
            else:
                st.error("Incorrect password")


# ========== PAGE 2 ==========
if st.session_state.page == 2:
    show_custom_loader()
    transition_effect()

    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("kcet_logo.png", width=60)
    with col2:
        st.title("Assistant Pre-check")

    play_welcome_audio()

    st.markdown("""
        <marquee behavior="scroll" direction="left" style="color:red; font-size:16px;">
        ⚠️ Make sure your email is correct to receive the exported chat logs.
        </marquee>
    """, unsafe_allow_html=True)

    email_to = st.text_input("📧 Enter your email to receive export")
    if email_to:
        st.session_state.export_email = email_to

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➡️ Start Chat", use_container_width=True):
            st.session_state.page = 3
            st.rerun()
    with col2:
        if st.button("⬅️ Back", use_container_width=True):
            if "user_email" in st.session_state and "username" in st.session_state:
                name = st.session_state.username
                email = st.session_state.user_email
                phone = st.session_state.get("user_phone", "Not provided")

                logout_message = f"""Hi {name},

You have logged out from KCET Chatbot (via Back button from Page 2).

Details:
Name: {name}
Email: {email}
Phone: {phone}

Thanks!"""
                send_email(email, "KCET Chatbot - Logout Confirmation", logout_message)
            st.session_state.page = 1
            st.rerun()

# ========== PAGE 3 ==========
if st.session_state.page == 3:
    show_custom_loader()

if st.session_state.page == 3:
    transition_effect()
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("kcet_logo.png", width=60)
    with col2:
        st.title("Chat with KCET Bot")

    with st.sidebar:
        theme_label = st.radio("🎨 Select Chat Theme", ["Modern Blue", "Mint Green", "Elegant Gold"])
        lang_choice = st.radio("🌐 Select Language", ["English", "Tamil"])
        st.session_state.language = "ta" if lang_choice == "Tamil" else "en"

    themes = {
        "Modern Blue": ("#e1ecf4", "#000000"),
        "Mint Green": ("#d2f4e3", "#000000"),
        "Elegant Gold": ("#fff4cc", "#000000")
    }

    user_color, _ = themes[theme_label]

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your message")
        send = st.form_submit_button("Send")

    if send and user_input:
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

    # ✅ CHAT DISPLAY with BLACK TEXT
    for speaker, msg, role in st.session_state.original_log:
        align = 'right' if role == "User" else 'left'
        bubble_color = user_color if role == "User" else "#f0f0f0"
        st.markdown(f"""
            <div style='background-color:{bubble_color}; padding:10px; margin:10px; border-radius:10px; text-align:{align}; color: black; transition: 0.5s all;'>
                <b>{speaker}</b>: {msg}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # 📥 Export to PDF & Download
    with col1:
        if st.button("📤 Export to PDF", use_container_width=True):
            pdf_path = export_pdf_from_log()
            st.success("PDF exported!")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="kcet_chat.pdf")

    # 📧 Email PDF
    with col2:
        if st.button("📧 Email PDF to Me", use_container_width=True):
            pdf_path = export_pdf_from_log()
            user_email = st.session_state.get("export_email", None)
            if user_email:
                try:
                    send_email(
                        to_email=user_email,
                        subject="KCET Chatbot Chat Log PDF",
                        body=f"Hi {st.session_state.username},\n\nPlease find attached the PDF export of your KCET chatbot session.",
                        attachment=pdf_path
                    )
                    st.success(f"PDF emailed to {user_email}!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
            else:
                st.warning("Please enter your email on the previous page (Assistant Pre-check).")

    # 🗑️ Clear Chat
    with col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.original_log = []
            st.rerun()

    # 🔒 Logout with Email
    with col4:
        if st.button("🔒 Logout", use_container_width=True):
            if "user_email" in st.session_state and "username" in st.session_state:
                name = st.session_state.username
                email = st.session_state.user_email
                phone = st.session_state.get("user_phone", "Not provided")

                logout_message = f"""Hi {name},

You have successfully logged out from KCET Chatbot.

Details:
Name: {name}
Email: {email}
Phone: {phone}

Thanks!"""

                try:
                    send_email(email, "KCET Chatbot - Logout Confirmation", logout_message)
                    st.success("Logout mail sent.")
                except Exception as e:
                    st.warning(f"Email send failed: {e}")

            st.session_state.page = 1
            st.rerun()

# ========== PAGE 4 ==========
if st.session_state.page == 4:
    show_custom_loader()
    transition_effect()

    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("kcet_logo.png", width=60)
    with col2:
        st.title("🛠️ Admin Dashboard")

    conn = sqlite3.connect(DB_FILE)
    users = pd.read_sql_query("SELECT * FROM users", conn)
    logs = pd.read_sql_query("SELECT * FROM chatlog", conn)
    conn.close()

    st.subheader("👥 Registered Users")
    st.dataframe(users)

    st.subheader("📋 Chat Logs")
    st.dataframe(logs)

    st.subheader("📈 Chat Analytics")
    user_msgs = logs[logs['role'] == "User"]
    bot_msgs = logs[logs['role'] == "Assistant"]
    col_u, col_b = st.columns(2)
    with col_u:
        st.markdown("**User Messages Count**")
        st.bar_chart(user_msgs['username'].value_counts())
    with col_b:
        st.markdown("**Assistant Messages Count**")
        st.bar_chart(bot_msgs['username'].value_counts())

    if st.button("📧 Email Logs", use_container_width=True):
        csv_path = os.path.join(tempfile.gettempdir(), "kcet_logs.csv")
        logs.to_csv(csv_path, index=False)
        send_email(SENDER_EMAIL, "KCET Logs", "Attached are all chat logs.", csv_path)
        st.success("Logs emailed to admin")

    if st.button("🏠 Back to Main Page", use_container_width=True):
        st.session_state.page = 1
        st.rerun()
