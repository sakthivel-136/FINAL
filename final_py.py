import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import time
import base64
import smtplib
import requests
from datetime import datetime
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# --- Config ---
tf_vector_file = "vectorized.pkl"
csv_file = "kcet.csv"
threshold = 0.6
sender_email = "kamarajengg.edu.in@gmail.com"
sender_password = "your_app_password_here"

# --- Streamlit Page ---
st.set_page_config(page_title="KCET Chatbot", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    user_name = st.text_input("Your name", placeholder="e.g., Shakthivel")
    mode = st.radio("Select Theme", ["Dark", "Light"], index=0)
    export_option = st.checkbox("Enable Export Options")
    email = st.text_input("📧 Enter your email address")
    file_type = st.radio("Export format", ["PDF", "TXT", "DOC"], index=0)
    sms_number = st.text_input("📱 Enter your mobile number for SMS (10 digits only)", max_chars=10, placeholder="9876543210")

is_dark = mode == "Dark"
bg_color = "#111" if is_dark else "#fff"
txt_color = "white" if is_dark else "black"

# --- Custom CSS ---
st.markdown(f"""
<style>
.scrolling-banner {{
    overflow: hidden;
    white-space: nowrap;
    animation: scroll-left 20s linear infinite;
    color: gold;
    background-color: {bg_color};
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
    color: {txt_color};
    text-align: center;
    padding: 5px 0 5px 0;
    font-weight: bold;
}}
</style>
<div class="chat-header">🏛️ KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY</div>
<div class="scrolling-banner">
    💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
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

# --- Chat Session State ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]

# --- Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your question here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Chat Logic ---
if submitted and user_input.strip():
    st.session_state.chat_log.append((user_name or "You", user_input.strip(), "User"))
    vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()
    base_response = df.iloc[idx]['Answer'] if max_sim >= threshold else "❌ Sorry, I couldn't understand that. Please rephrase."
    st.session_state.chat_log.append(("KCET Assistant", base_response, "Assistant"))
    st.rerun()

# --- Display Chat ---
for speaker, msg, role in st.session_state.chat_log:
    align = 'right' if speaker == (user_name or "You") else 'left'
    bg = "#d0e8f2" if align == 'right' else "#d1d1e9"
    st.markdown(f"""
    <div style='background-color:{bg}; text-align:{align}; padding:10px; border-radius:10px; margin:5px 0;'>
        <b>{speaker}</b> ({role}): {msg}
    </div>
    """, unsafe_allow_html=True)

# --- Email Sending Function ---
def send_email(recipient_email, subject, body, attachment_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg.set_content(body)
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=os.path.basename(attachment_path))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        return str(e)

# --- SMS Sending Function ---
def send_free_sms(to_number, message_text):
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {
        'authorization': 'UewB056c7fzvRHWoKFCSxXpirVuIAts1ldjYqObQmEZJN42TaMiOVM0A8aenGHURofDdCT2Xc5wuBQmJ',
        'Content-Type': "application/x-www-form-urlencoded"
    }
    payload = {
        'sender_id': 'FSTSMS',
        'message': message_text,
        'language': 'english',
        'route': 'v3',
        'numbers': to_number
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

# --- Export Button ---
if export_option:
    st.subheader("📤 Export Chat")
    if st.button("Download / Email"):
        try:
            filename = f"{user_name}_chatlog.{file_type.lower()}"
            if file_type == "PDF":
                pdf = FPDF()
                pdf.add_page()
                if os.path.exists("kcet_logo.png"):
                    pdf.image("kcet_logo.png", x=10, y=8, w=20, h=20)
                pdf.set_font("Arial", 'B', 16)
                pdf.set_xy(35, 10)
                pdf.cell(160, 10, txt="KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY", ln=True, align="L")
                pdf.set_font("Arial", '', 11)
                pdf.set_xy(10, 25)
                banner_text = "💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration"
                pdf.multi_cell(0, 8, banner_text)
                pdf.line(10, 38, 200, 38)
                pdf.set_xy(10, 45)
                pdf.set_font("Arial", '', 12)
                for speaker, msg, role in st.session_state.chat_log:
                    msg_clean = msg.replace('\xa0', ' ')
                    pdf.multi_cell(0, 10, f"{speaker} ({role}): {msg_clean}")
                pdf_data = pdf.output(dest='S').encode('latin-1')
                mime = "application/pdf"
                data = pdf_data
            else:
                text = "\n".join([f"{s} ({r}): {m}" for s, m, r in st.session_state.chat_log])
                data = text.encode("utf-8", "ignore")
                mime = "application/msword" if file_type == "DOC" else "text/plain"

            st.download_button("📥 Download", data=data, file_name=filename, mime=mime)

            if email and "@" in email:
                with open(filename, "wb") as f:
                    f.write(data)
                sent = send_email(email, f"{user_name} Chat Log", "Attached is your chat log.", filename)
                if sent is True:
                    st.success("✅ Email sent!")
                else:
                    st.error(f"❌ Email error: {sent}")

            if sms_number and len(sms_number) == 10 and sms_number.isdigit():
                sms_response = send_free_sms(sms_number, f"Hello {user_name}, your KCET Chat PDF has been emailed. - KCET Assistant")
                if sms_response.get("return") == True:
                    st.success("📲 SMS sent successfully!")
                else:
                    st.error(f"SMS failed: {sms_response}")

        except Exception as e:
            st.error(f"❌ Export failed: {e}")

# --- Clear Chat ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]
    st.rerun()
