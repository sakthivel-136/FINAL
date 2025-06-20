import streamlit as st
import pandas as pd
import pickle
import os
import uuid
import time
import base64
import smtplib
import json
from datetime import datetime
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# --- Config ---
tf_vector_file = "vectorized.pkl"
csv_file = "kcet.csv"
threshold = 0.6
sender_email = ("kamarajengg.edu.in@Gmail.com")
sender_password = ("vwvc wsff fbrv umzh ")
profile_file = "user_profile.json"

# --- Streamlit Page ---
st.set_page_config(page_title="KCET Chatbot", layout="centered")

# --- Load Profile ---
def load_profile():
    if os.path.exists(profile_file):
        with open(profile_file, "r") as f:
            return json.load(f)
    return {
        "name": "Guest",
        "avatar": "https://cdn-icons-png.flaticon.com/512/2922/2922506.png",
        "role": "Student",
        "color": "#d0e8f2",
        "text_color": "#000"
    }

# --- Save Profile ---
def save_profile(profile):
    with open(profile_file, "w") as f:
        json.dump(profile, f)

if "user_profile" not in st.session_state:
    st.session_state.user_profile = load_profile()

# --- Sidebar ---
with st.sidebar:
    st.image(st.session_state.user_profile["avatar"], width=100)
    st.title("⚙️ Settings")
    st.text_input("Your Name", value=st.session_state.user_profile["name"], key="user_name")
    uploaded_avatar = st.file_uploader("Upload Avatar", type=["png", "jpg", "jpeg"])
    st.selectbox("Select Role", ["Student", "Staff", "Faculty"], index=["Student", "Staff", "Faculty"].index(st.session_state.user_profile["role"]), key="user_role")
    user_bubble_color = st.color_picker("Bubble Color", value=st.session_state.user_profile.get("color", "#d0e8f2"))
    user_text_color = st.color_picker("Text Color", value=st.session_state.user_profile.get("text_color", "#000"))
    if st.button("Update Profile"):
        if uploaded_avatar:
            avatar_path = f"avatar_{uuid.uuid4().hex}.png"
            with open(avatar_path, "wb") as f:
                f.write(uploaded_avatar.read())
            st.session_state.user_profile["avatar"] = avatar_path
        st.session_state.user_profile["name"] = st.session_state.user_name
        st.session_state.user_profile["role"] = st.session_state.user_role
        st.session_state.user_profile["color"] = user_bubble_color
        st.session_state.user_profile["text_color"] = user_text_color
        save_profile(st.session_state.user_profile)
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

if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hi there! 👋 I'm the KCET Assistant. How can I help you today?", "Assistant")]

# --- Input Form ---
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Type your question here...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

# --- Chat Logic ---
if submitted and user_input.strip():
    user_role = st.session_state.user_profile["role"]
    st.session_state.chat_log.append((st.session_state.user_profile["name"], user_input.strip(), user_role))

    vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()

    base_response = df.iloc[idx]['Answer'] if max_sim >= threshold else "❌ Sorry, I couldn't understand that. Please rephrase."
    if user_role == "Faculty":
        full_response = base_response + "\n👩‍🏫 As Faculty, you can reach academic research support at research@kcet.ac.in."
    elif user_role == "Staff":
        full_response = base_response + "\n🧑‍💼 Staff members can view administrative resources on the intranet."
    else:
        full_response = base_response

    st.session_state.chat_log.append(("KCET Assistant", full_response, "Assistant"))
    st.rerun()

# --- Display Chat ---
st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
for speaker, msg, role in st.session_state.chat_log:
    align = 'right' if speaker == st.session_state.user_profile["name"] else 'left'
    avatar = st.session_state.user_profile["avatar"] if speaker == st.session_state.user_profile["name"] else "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
    if speaker == st.session_state.user_profile["name"]:
        bg = st.session_state.user_profile["color"]
        txt = st.session_state.user_profile["text_color"]
    else:
        bg = "#d1d1e9"
        txt = "#000"
    safe_msg = msg.encode("ascii", errors="ignore").decode("ascii")
    st.markdown(f"""
    <div class='message' style='background-color:{bg}; text-align:{align}; color:{txt};'>
        <img src='{avatar}' class='avatar'/>
        <div><b>{speaker}</b> ({role}): {safe_msg}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Export Button ---
if export_option:
    st.subheader("📤 Export Chat")
    email = st.text_input("Email Address")
    if st.button("Send PDF Report"):
        try:
            filename = f"kcet_chat_{uuid.uuid4().hex}.pdf"
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)

            pdf.cell(200, 10, txt="KCET Assistant Chat Log", ln=True, align="C")
            pdf.ln(5)

            for speaker, msg, role in st.session_state.chat_log:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                msg_clean = msg.encode("ascii", errors="ignore").decode("ascii")
                pdf.multi_cell(0, 10, f"[{timestamp}] {speaker} ({role}): {msg_clean}")

            pdf.output(filename)

            msg = EmailMessage()
            msg['Subject'] = "KCET Chat Log"
            msg['From'] = sender_email
            msg['To'] = email
            msg.set_content("Here is your chat log with the KCET Assistant.")

            with open(filename, "rb") as f:
                msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=filename)

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(sender_email, sender_password)
                smtp.send_message(msg)

            st.success("✅ Email sent successfully!")
            os.remove(filename)
        except Exception as e:
            st.error(f"❌ Email Error: {e}")

# --- Clear Chat ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("KCET Assistant", "Hi there! 👋 I'm the KCET Assistant. How can I help you today?", "Assistant")]
    st.rerun()
