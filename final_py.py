# KCET Chatbot: Full Code with Page 1, 2, 3 and Admin Dashboard
import streamlit as st
import base64, os, pickle, re, tempfile, time, sqlite3
import pandas as pd
from gtts import gTTS
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from docx import Document
from docx.shared import Pt
import openai

# ========== CONFIG ==========
openai.api_key = "sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl"
ADMIN_PASSWORD = "kcetadmin123"
TFIDF_FILE = "vectorized.pkl"
CSV_FILE = "kcet.csv"
DB_FILE = "kcet_chatlog.db"
THRESHOLD = 0.6

# ========== INITIAL STATE ==========
def init_state():
    defaults = {
        "page": 1, "img_idx": 0, "autoplay_enabled": True, "music_played": False,
        "language": "en", "original_log": [], "last_input": "",
        "username": "You", "user_color": "#d0e8f2", "bot_color": "#d1d1e9",
        "enable_export": True, "admin_page": False
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
    conn.commit()
    conn.close()

def save_to_db(user, role, msg):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO chatlog (username, role, message) VALUES (?, ?, ?)", (user, role, msg))
    conn.commit()
    conn.close()

init_db()

# ========== UTILS ==========
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def speak_text(text):
    lang = 'ta' if st.session_state.language == 'ta' else 'en'
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        audio = base64.b64encode(open(f.name, "rb").read()).decode()
        st.markdown(f"<audio autoplay><source src='data:audio/mp3;base64,{audio}' type='audio/mp3'></audio>", unsafe_allow_html=True)

@st.cache_data

def load_vector_data():
    if os.path.exists(TFIDF_FILE):
        with open(TFIDF_FILE, "rb") as f:
            return pickle.load(f)
    else:
        df = pd.read_csv(CSV_FILE)
        df['Question'] = df['Question'].str.lower().str.strip()
        vec = TfidfVectorizer()
        X = vec.fit_transform(df['Question'])
        with open(TFIDF_FILE, "wb") as f:
            pickle.dump((vec, X, df), f)
        return vec, X, df


vectorizer, vectors, df = load_vector_data()

def get_gpt_response(prompt):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for KCET college students."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        return res.choices[0].message.content
    except:
        return "⚠️ GPT is currently unavailable. Please try again later."

# ========== PAGE 1: Welcome ==========
if st.session_state.page == 1:
    st.set_page_config(page_title="KCET Welcome", layout="centered")
    if os.path.exists("kcet_logo.png"):
        logo_b64 = base64.b64encode(open("kcet_logo.png", "rb").read()).decode()
        st.markdown(f"""
        <div style='text-align:center;'>
            <img src='data:image/png;base64,{logo_b64}' style='height:100px;width:100px;border-radius:50%;'>
            <h2>KAMARAJ COLLEGE OF ENGINEERING AND TECHNOLOGY</h2>
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.music_played and os.path.exists("kcet_music.mp3"):
        music_b64 = base64.b64encode(open("kcet_music.mp3", "rb").read()).decode()
        st.markdown(f"<audio autoplay loop><source src='data:audio/mp3;base64,{music_b64}'></audio>", unsafe_allow_html=True)
        st.session_state.music_played = True

    images = [f for f in os.listdir("college_images") if f.lower().endswith(('png', 'jpg'))]
    captions = [f.replace("_", " ").split(".")[0] for f in images]
    idx = st.session_state.img_idx
    col1, col2, col3 = st.columns([1, 6, 1])
    if col1.button("⟨"):
        st.session_state.img_idx = (idx - 1) % len(images)
    if col3.button("⟩"):
        st.session_state.img_idx = (idx + 1) % len(images)
    if images:
        st.image(Image.open(os.path.join("college_images", images[idx])), use_container_width=True)
        st.caption(captions[idx])

    st.divider()
    st.markdown("### 🔐 Admin Dashboard")
    if st.checkbox("Login as Admin"):
        password = st.text_input("Enter Password", type="password")
        if password == ADMIN_PASSWORD:
            st.success("✅ Admin access granted")
            conn = sqlite3.connect(DB_FILE)
            logs = pd.read_sql("SELECT * FROM chatlog ORDER BY timestamp DESC", conn)
            st.dataframe(logs)
            conn.close()
        elif password:
            st.error("❌ Incorrect password")

    st.divider()
    if st.button("Go to Chatbot"):
        st.session_state.page = 99
        st.rerun()

# ========== PAGE 2: Loader ==========
elif st.session_state.page == 99:
    st.set_page_config(page_title="Launching KCET Chatbot", layout="centered")
    st.markdown("""
        <div style='text-align:center; margin-top:100px;'>
            <img src='https://i.gifer.com/VAyR.gif' width='100'>
            <h4 style='margin-top:20px;'>Launching KCET Chatbot...</h4>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.page = 2
    st.rerun()

# ========== PAGE 3: Chatbot ==========
elif st.session_state.page == 2:
    st.set_page_config(page_title="KCET Chatbot", layout="centered")
    if st.button("🏠 Home"):
        st.session_state.page = 1
        st.rerun()

    st.markdown("""
        <div style='text-align:center;'>
            <h3>🤖 KCET Chatbot is now active!</h3>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ Settings")
    st.session_state.username = st.sidebar.text_input("🧑 Your Name", st.session_state.username)
    st.session_state.user_color = st.sidebar.color_picker("User Color", st.session_state.user_color)
    st.session_state.bot_color = st.sidebar.color_picker("Bot Color", st.session_state.bot_color)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question", value=st.session_state.last_input)
        submitted = st.form_submit_button("Send")
        if submitted:
            st.session_state.last_input = ""

    if submitted and user_input.strip():
        msg = remove_emojis(user_input.strip())
        st.session_state.original_log.append((st.session_state.username, msg, "User"))
        save_to_db(st.session_state.username, "User", msg)

        vec = vectorizer.transform([msg.lower()])
        similarity = cosine_similarity(vec, vectors)
        idx = similarity.argmax()
        max_sim = similarity.max()

        if max_sim >= THRESHOLD:
            answer = df.iloc[idx]['Answer']
        else:
            answer = get_gpt_response(msg)

        answer = remove_emojis(answer)
        if st.session_state.language == "ta":
            answer = GoogleTranslator(source='en', target='ta').translate(answer)

        st.session_state.original_log.append(("KCET Assistant", answer, "Assistant"))
        save_to_db("KCET Assistant", "Assistant", answer)

        with st.spinner("KCET Assistant typing..."):
            time.sleep(min(1.5, len(answer)/20))
            speak_text(answer)

    for speaker, msg, role in st.session_state.original_log:
        align = 'right' if role == "User" else 'left'
        color = st.session_state.user_color if role == "User" else st.session_state.bot_color
        st.markdown(f"""
            <div style='background-color:{color}; padding:10px; margin:10px; border-radius:10px; text-align:{align};'>
                <b>{speaker}</b>: {msg}
            </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌐 Translate to Tamil" if st.session_state.language == "en" else "🌐 Back to English"):
            st.session_state.language = "ta" if st.session_state.language == "en" else "en"
            for i in range(len(st.session_state.original_log)):
                s, m, r = st.session_state.original_log[i]
                if r == "Assistant":
                    st.session_state.original_log[i] = (s, GoogleTranslator(source='en' if st.session_state.language == 'ta' else 'ta', target=st.session_state.language).translate(m), r)
            st.rerun()

    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.original_log.clear()
            st.rerun()
