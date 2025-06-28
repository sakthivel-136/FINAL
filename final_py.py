
import streamlit as st
import base64, os
from PIL import Image

st.set_page_config(page_title="KCET Welcome", layout="centered")

# College logo
with open("kcet_logo.png", "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()
st.markdown(f"""
    <div style='text-align:center;'>
        <img src='data:image/png;base64,{logo_base64}' style='height:100px; border-radius:50%;'>
        <h2>KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY</h2>
    </div>
""", unsafe_allow_html=True)

# Image slider
folder = "college_images"
images = [img for img in os.listdir(folder) if img.endswith((".png", ".jpg"))]
idx = st.session_state.get("img_idx", 0)

cols = st.columns([1, 6, 1])
with cols[0]:
    if st.button("◀", key="prev"): st.session_state.img_idx = (idx - 1) % len(images)
with cols[1]:
    img = Image.open(os.path.join(folder, images[idx]))
    st.image(img, use_column_width=True, caption=images[idx].split(".")[0].replace("_", " "))
with cols[2]:
    if st.button("▶", key="next"): st.session_state.img_idx = (idx + 1) % len(images)

# Navigation buttons
if st.button("Go to Chatbot"): st.switch_page("page3.py")
if st.button("Admin Panel"):
    st.session_state.show_login = True
    st.switch_page("page4.py")


## ✅ Page 2: Loader Page


# page2.py
import streamlit as st
import time

st.set_page_config(page_title="Loading KCET Chatbot")
st.markdown("""
    <div style='text-align:center; margin-top:100px;'>
        <img src='https://i.gifer.com/VAyR.gif' width='100'>
        <h4>Launching KCET Chatbot...</h4>
    </div>
""", unsafe_allow_html=True)
time.sleep(2)
st.switch_page("page3.py")


## ✅ Page 3: Chatbot Page


import streamlit as st, pandas as pd, re, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import sqlite3
from docx import Document
from fpdf import FPDF

st.set_page_config(page_title="KCET Chatbot")

st.sidebar.title("Settings")
name = st.sidebar.text_input("Your Name", value="Student")
email = st.sidebar.text_input("Your Email")
user_color = st.sidebar.color_picker("User Bubble", "#d0e8f2")
bot_color = st.sidebar.color_picker("Bot Bubble", "#d1d1e9")
language = st.sidebar.selectbox("Language", ["en", "ta"])

st.session_state.setdefault("log", [])

# Load CSV
df = pd.read_csv("kcet.csv")
df['Question'] = df['Question'].str.lower()
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['Question'])

# Chat input
query = st.text_input("Ask a question...")
if st.button("Send") and query:
    cleaned = re.sub(r'[^\x00-\x7F]+', '', query)
    vec = vectorizer.transform([cleaned.lower()])
    score = cosine_similarity(vec, vectors).flatten()
    idx = score.argmax()
    ans = df.iloc[idx]['Answer'] if score[idx] > 0.6 else "Sorry, I couldn't understand that."
    if language == 'ta':
        ans = GoogleTranslator(source='en', target='ta').translate(ans)
    st.session_state.log.append((name, cleaned, "User"))
    st.session_state.log.append(("KCET Bot", ans, "Bot"))

# Display chat
for speaker, msg, role in st.session_state.log:
    color = user_color if role == "User" else bot_color
    align = "right" if role == "User" else "left"
    st.markdown(f"""
        <div style='text-align:{align}; background-color:{color}; padding:10px; margin:5px; border-radius:10px;'>
            <b>{speaker}</b>: {msg}
        </div>
    """, unsafe_allow_html=True)

# Save to SQLite
conn = sqlite3.connect("kcet_chatlog.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chatlog (
        username TEXT, role TEXT, message TEXT, email TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
for speaker, msg, role in st.session_state.log:
    cursor.execute("INSERT INTO chatlog (username, role, message, email) VALUES (?, ?, ?, ?)", (speaker, role, msg, email))
conn.commit()

# Export options
st.subheader("Export Chat")
export_type = st.radio("Format", ["DOCX", "PDF"])
if st.button("Download"):
    if export_type == "DOCX":
        doc = Document()
        doc.add_heading("KCET Chat Export", 0)
        for speaker, msg, role in st.session_state.log:
            doc.add_paragraph(f"{speaker} ({role}): {msg}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("Download DOCX", f, file_name="chat.docx")
    else:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for speaker, msg, role in st.session_state.log:
            pdf.cell(200, 10, txt=f"{speaker} ({role}): {msg}", ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("Download PDF", f, file_name="chat.pdf")


## ✅ Page 4: Admin Dashboard


import streamlit as st, sqlite3, pandas as pd
from fpdf import FPDF
import smtplib
from email.message import EmailMessage

st.set_page_config(page_title="Admin Dashboard")

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    pwd = st.text_input("Enter Admin Password", type="password")
    if st.button("Login"):
        if pwd == "qwerty12345":
            st.session_state.admin_authenticated = True
        else:
            st.error("Incorrect password")
else:
    st.title("Admin Panel")
    conn = sqlite3.connect("kcet_chatlog.db")
    df = pd.read_sql_query("SELECT * FROM chatlog", conn)
    st.dataframe(df)

    if st.button("Export All as PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for _, row in df.iterrows():
            pdf.cell(200, 10, txt=f"{row['username']} ({row['role']}): {row['message']}", ln=True)
        path = "admin_export.pdf"
        pdf.output(path)
        with open(path, "rb") as f:
            st.download_button("Download PDF", f, file_name=path)

    if st.button("Email Export to Admin"):
        email = "kamarajengg.edu.in@gmail.com"
        msg = EmailMessage()
        msg["Subject"] = "KCET Admin Chat Export"
        msg["From"] = email
        msg["To"] = email
        msg.set_content("Attached is the full KCET chat log.")
        with open("admin_export.pdf", "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="chat_log.pdf")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(email, "your_app_password")
            smtp.send_message(msg)
        st.success("✅ PDF exported and emailed successfully!")

