import streamlit as st
import pandas as pd
import pickle
import os
import smtplib
import re
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import base64
from deep_translator import GoogleTranslator
import time

# ========== EMAIL CREDENTIALS ==========
SENDER_EMAIL = "kamarajengg.edu.in@gmail.com"
SENDER_PASSWORD = "vwvcwsfffbrvumzh"  # Gmail App Password

# ========== Constants ==========
tf_vector_file = "vectorized.pkl"
csv_file = "kcet.csv"
threshold = 0.6

# ========== Remove Emojis ==========
def remove_emojis(text):
    return re.sub(r'[^
