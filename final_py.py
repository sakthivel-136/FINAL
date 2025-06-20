import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import smtplib
from datetime import datetime
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# --------------------------
# ✅ Email Credentials
# --------------------------
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = ""vwvc wsff fbrv umzh  # Use Gmail App Password only

# ✅ Reusable Email Function
def send_email(recipient_email, subject, body, attachment_path):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg.set_content(body)

    try:
        with open(attachment_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename=os.path.basename(attachment_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        return str(e)

# --------------------------
# Chatbot Config
# --------------------------
tf_vector_file = "vectorized.pkl"
csv_file = "kcet.csv"
threshold = 0.6

st.set_page_config(page_title="KCET Chatbot", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio("Select Theme", ["Dark", "Light"], index=0)
    export_option = st.checkbox("Enable Export Options")

is_dark = mode == "Dark"
bg_color = "#111" if is_dark else "#fff"
txt_color = "white" if is_dark else "black"

# --- CSS and Banner ---
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

# --- Chat State ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]

# --- Chat Input ---
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
    st.session_state.chat_log.append(("KCET Assistant", base_response, "Assistant"))
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

# --- Export Section ---
if export_option:
    st.subheader("📤 Export Chat")
    file_type = st.radio("Choose file type", ["PDF", "TXT", "DOC"], index=0)
    email = st.text_input("📧 Email (optional)", placeholder="example@gmail.com")

    if st.button("Generate and Download"):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"kcet_chat_{timestamp}.{file_type.lower()}"

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
                file_data = pdf.output(dest='S').encode('latin-1')
                download_data = file_data
                mime = "application/pdf"
            else:
                text_data = ""
                for speaker, msg, role in st.session_state.chat_log:
                    msg_clean = msg.replace('\xa0', ' ')
                    text_data += f"{speaker} ({role}): {msg_clean}\n"
                file_data = text_data.encode("utf-8")
                download_data = file_data
                mime = "application/msword" if file_type == "DOC" else "text/plain"

            # ✅ Save for emailing
            with open(filename, "wb") as f:
                f.write(download_data)

            # ✅ Download
            st.download_button(
                label=f"📥 Download {file_type}",
                data=download_data,
                file_name=filename,
                mime=mime
            )

            # ✅ Send email if email is valid
            if email and "@" in email:
                subject = "KCET Assistant Chat Log"
                body = "Please find the attached KCET Assistant chat log."
                result = send_email(email, subject, body, filename)
                if result == True:
                    st.success("✅ Email sent successfully!")
                else:
                    st.error(f"❌ Failed to send email: {result}")
            elif email:
                st.warning("⚠️ Invalid email format.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]
    st.rerun()
