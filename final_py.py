import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from gtts import gTTS
import tempfile
import base64

# =============== Voice Output ===============
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_base64 = base64.b64encode(open(fp.name, "rb").read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# =============== Constants ===============
tf_vector_file = "vectorized.pkl"
csv_file = "kcet.csv"
threshold = 0.6

st.set_page_config(page_title="KCET Chatbot", layout="centered")

# =============== Sidebar ===============
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio("Theme", ["Dark", "Light"], index=0)
    is_dark = mode == "Dark"
    st.session_state.user_name = st.text_input("👤 Your Name", value=st.session_state.get("user_name", "Shakthivel"))
    user_bubble_color = st.color_picker("🎨 User Bubble", "#d0e8f2")
    assistant_bubble_color = st.color_picker("🎨 Assistant Bubble", "#d1d1e9")
    txt_color = st.color_picker("🖋️ Text Color", "#000000")
    export_option = st.checkbox("📤 Enable Export")

bg_color = "#111" if is_dark else "#fff"
final_txt_color = "white" if is_dark else txt_color  # ✅ Fixed logic
user_name = st.session_state.user_name

# =============== Header ===============
try:
    with open("kcet_logo.png", "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode()
except FileNotFoundError:
    encoded_img = ""

st.markdown(f"""
<style>
.circle-img {{
    border-radius: 50%;
    width: 50px;
    height: 50px;
    vertical-align: middle;
    margin-right: 10px;
}}
.title-text {{
    font-size: 22px;
    font-weight: bold;
    color: {final_txt_color};
    display: inline-block;
    vertical-align: middle;
}}
.header-container {{
    text-align: center;
    margin-top: 10px;
}}
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
    color: {final_txt_color};
    text-align: center;
    padding: 10px 0;
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
<div class="header-container">
    <img src="data:image/png;base64,{encoded_img}" class="circle-img">
    <span class="title-text">KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY</span>
</div>
<div class="scrolling-banner">
    💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration
</div>
<div class="chat-header">KCET Assistant</div>
""", unsafe_allow_html=True)

# =============== Load Vector Data ===============
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

# =============== Session State ===============
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]

# =============== Chat Form ===============
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    user_input = col1.text_input("Ask your question...", label_visibility="collapsed")
    submitted = col2.form_submit_button("➤")

if submitted and user_input.strip():
    st.session_state.chat_log.append((user_name, user_input.strip(), "User"))
    vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(vec, vectors)
    max_sim = similarity.max()
    idx = similarity.argmax()
    base_response = df.iloc[idx]['Answer'] if max_sim >= threshold else "❌ Sorry, I couldn't understand that. Please rephrase."
    st.session_state.chat_log.append(("KCET Assistant", base_response, "Assistant"))
    speak_text(base_response)
    st.rerun()

# =============== Chat Display ===============
st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
for speaker, msg, role in st.session_state.chat_log:
    align = 'right' if role == "User" else 'left'
    bg = user_bubble_color if role == "User" else assistant_bubble_color
    st.markdown(f"""
    <div class='message' style='background-color:{bg}; text-align:{align}; color:{final_txt_color};'>
        <div><b>{speaker}</b> ({role}): {msg.replace('\xa0', ' ')}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============== Export Chat ===============
if export_option:
    st.subheader("📤 Export Chat")
    file_type = st.radio("File Type", ["PDF", "TXT", "DOC"], index=0)

    if st.button("Download"):
        try:
            filename = f"{user_name}_chatlog.{file_type.lower()}"
            if file_type == "PDF":
                pdf = FPDF()
                pdf.add_page()
                if os.path.exists("kcet_logo.png"):
                    pdf.image("kcet_logo.png", x=10, y=8, w=20, h=20)
                pdf.set_font("Arial", 'B', 16)
                pdf.set_xy(35, 10)
                pdf.cell(160, 10, txt="KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.set_xy(10, 25)
                pdf.multi_cell(0, 8, "💼 100% Placement | 👩‍🏫 Top Faculty | 🎓 Research Driven | 🧠 Hackathons | 🤝 Industry Collaboration")
                pdf.line(10, 38, 200, 38)
                pdf.set_xy(10, 45)
                pdf.set_font("Arial", '', 12)
                for speaker, msg, role in st.session_state.chat_log:
                    pdf.multi_cell(0, 10, f"{speaker} ({role}): {msg.replace('\xa0', ' ')}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 Download", f.read(), file_name=filename, mime="application/pdf")
            else:
                text = "\n".join([f"{s} ({r}): {m}" for s, m, r in st.session_state.chat_log])
                mime = "application/msword" if file_type == "DOC" else "text/plain"
                st.download_button("📥 Download", text.encode(), file_name=filename, mime=mime)
        except Exception as e:
            st.error(f"❌ Export failed: {e}")

# =============== Clear Chat ===============
if st.button("🧹 Clear Chat"):
    st.session_state.chat_log = [("KCET Assistant", "Hello! I'm your KCET Assistant. Ask me anything.", "Assistant")]
    st.rerun()
