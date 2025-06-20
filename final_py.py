import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import io
import smtplib
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
sender_password = "vwvc wsff fbrv umzh"  # 🔑 Your app password here

# --- Streamlit Page ---
st.set_page_config(page_title="KCET Chatbot", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio("Select Theme", ["Dark", "Light"], index=0)
    export_option = st.checkbox("Enable Export Options")

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
    padding: 10px 0 5px 0;
    font-weight: bold;
}}
.message {{
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    animation: fadein 0.5s;
}}
@keyframes fadein {{
    from {{opacity: 0; transform: translateY(10px);}}
    to {{opacity: 1; transform: translateY(0);}}
}}
</style>
<div class="scrolling-banner">
    💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
</div>
<div class="chat-header">KCET Assistant</div>
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

# --- Session State Init ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]

# --- Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your question here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Chat Logic ---
if submitted and user_input.strip():
    st.session_state.chat_log.append(("You", user_input.strip(), "User"))

    vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    base_response = df.iloc[idx]['Answer'] if max_sim >= threshold else "❌ Sorry, I couldn't understand that. Please rephrase."
    full_response = base_response

    st.session_state.chat_log.append(("KCET Assistant", full_response, "Assistant"))
    st.rerun()

# --- Display Chat ---
st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
for speaker, msg, role in st.session_state.chat_log:
    align = 'right' if speaker == "You" else 'left'
    bg = "#d0e8f2" if speaker == "You" else "#d1d1e9"
    txt = "#000"
    msg_clean = msg.replace('\xa0', ' ')
    st.markdown(f"""
    <div class='message' style='background-color:{bg}; text-align:{align}; color:{txt};'>
        <div><b>{speaker}</b> ({role}): {msg_clean}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Export + Download + Email ---
if export_option:
    st.subheader("📤 Export Options")
    file_type = st.radio("Choose file type", ["PDF", "TXT", "DOC"], index=0)
    email = st.text_input("📧 Email (optional)", placeholder="example@gmail.com")

    if st.button("Generate and Download"):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"kcet_chat_{timestamp}.{file_type.lower()}"
            safe_filename = ''.join(c if ord(c) < 128 else '_' for c in filename)

            if file_type == "PDF":
                pdf = FPDF()
                pdf.add_page()
                pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
                pdf.set_font("DejaVu", size=12)
                pdf.cell(200, 10, txt="KCET Assistant Chat Log", ln=True, align="C")
                pdf.ln(5)
                for speaker, msg, role in st.session_state.chat_log:
                    msg_clean = msg.replace('\xa0', ' ')
                    pdf.multi_cell(0, 10, f"{speaker} ({role}): {msg_clean}")
                pdf_buffer = io.BytesIO()
                pdf.output(pdf_buffer)
                file_data = pdf_buffer.getvalue()
                mime = "application/pdf"
            else:
                text_data = ""
                for speaker, msg, role in st.session_state.chat_log:
                    msg_clean = msg.replace('\xa0', ' ')
                    text_data += f"{speaker} ({role}): {msg_clean}\n"
                file_data = text_data
                mime = "text/plain" if file_type == "TXT" else "application/msword"

            # 🎯 Download to laptop
            st.download_button(
                label=f"📥 Download {file_type}",
                data=file_data,
                file_name=filename,
                mime=mime
            )

            # 📧 Send Email if email is given
            if email and "@" in email:
                msg = EmailMessage()
                msg['Subject'] = "KCET Assistant Chat Log"
                msg['From'] = sender_email
                msg['To'] = email
                msg.set_content("Please find the KCET Assistant chat log attached.")

                if isinstance(file_data, str):
                    file_data = file_data.encode("utf-8")
                msg.add_attachment(file_data, maintype="application", subtype=mime.split("/")[-1], filename=safe_filename)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(sender_email, sender_password)
                    smtp.send_message(msg)

                st.success("✅ Chat log sent via email!")
            elif email:
                st.warning("⚠️ Invalid email address.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Clear Chat ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]
    st.rerun()
