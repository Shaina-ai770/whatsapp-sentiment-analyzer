import streamlit as st

st.set_page_config(
    page_title='WhatsApp Sentiment Analyzer',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={'About': "WhatsApp Sentiment Analyzer – B.Tech Final Year Project 2025"}
)

import pandas as pd
from typing import Dict, Optional, Tuple
from src.parser import parse_chat
from src.preprocess import preprocess_df, clean_text
from src.sentiment import apply_vader
from src.visualize import save_wordcloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime
import numpy as np
import re
import emoji
from transformers import pipeline
import matplotlib.pyplot as plt
from nrclex import NRCLex
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import time
import logging
import torch
from fpdf import FPDF
from src.advanced_sentiment import AdvancedSentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

SUPPORTED_LANGUAGES = ["english", "hindi", "hinglish", "urdu", "spanish", "french", "german", "arabic", "russian", "chinese", "portuguese", "italian", "japanese", "korean"]


# ══════════════════════════════════════════════════════════════════════════════
#   MULTILINGUAL ENGINE — Pure XLM-RoBERTa (No Translation Needed)
#   State-of-the-art: cardiffnlp/twitter-xlm-roberta-base-sentiment
#   Trained on 100+ languages — Hindi, Urdu, Spanish, Arabic, Hinglish etc.
# ══════════════════════════════════════════════════════════════════════════════

# ── Hinglish Roman-script word lists (for detection + lexicon scoring) ────────
_HINDI_ROMAN_WORDS = {
    "hai","hain","nahi","nahin","kya","karo","mera","tera","bhai","yaar",
    "accha","acha","theek","kal","aaj","hua","toh","bhi","bahut","bohot",
    "zyada","sahi","bas","kuch","abhi","phir","woh","yeh","ye","wo",
    "main","mein","hum","aap","tum","unka","uska","iska","inhe","unhe",
    "liye","saath","matlab","samajh","dekh","sun","bata","bol","jaa",
    "aa","kar","ho","tha","thi","the","hoga","hogi","honge","chahiye",
    "kyun","kyunki","lekin","aur","ya","agar","isliye","tabhi","pehle",
    "baad","kab","kahan","kaisa","kaisi","kitna","kitni","pyaar","pyar",
    "dost","ghar","kaam","paise","din","raat","khana","paani","chai"
}

_HINGLISH_POSITIVE = {
    "accha","achha","badhiya","mast","zabardast","kamaal","sahi","khush",
    "khushi","maja","maza","bindaas","shandar","behtareen","wah","wahh",
    "superb","amazing","love","pyaar","dil","sundar","beautiful","happy",
    "excited","proud","shukriya","mubarak","badhai","top","best","solid",
    "ekdum","bilkul","zaroor","legend","hero","cute","sweet","caring",
    "helpful","great","nice","awesome","perfect","wonderful","brilliant",
    "zindagi","khubsoorat","umeed","hope","mazedaar","jabardast","toh accha",
    "bohot accha","bahut accha","achi","shukriya","dhanyawad","congratulations"
}

_HINGLISH_NEGATIVE = {
    "bura","buri","galat","ganda","bekar","bakwaas","faltu","nahi","na",
    "dard","takleef","pareshaan","tension","thaka","dukh","dukhi","rona",
    "gussa","naraaz","angry","sad","hurt","hate","nafrat","barbad","fail",
    "mushkil","darr","bimaar","jhooth","dhoka","maar","ladhna","jhagda",
    "problem","nuksaan","haarna","toota","toot","pareshan","bechaini",
    "bura laga","dukhi hun","roya","toot gaya","barbad ho gaya","gussa hai",
    "naraaz hun","bahut bura","bohot bura","bilkul galat","ekdum bekar"
}

_HINGLISH_INTENSIFIERS = {
    "bahut","bohot","ekdum","bilkul","poora","pura","itna","zyada","kaafi"
}


def detect_text_language(text):
    """
    Detect language with special Hinglish handling.
    Hinglish = Roman script Hindi mixed with English.
    """
    text_str = str(text).strip()
    if not text_str or len(text_str) < 3:
        return "english"

    words = set(re.findall(r'\b\w+\b', text_str.lower()))

    # Hinglish check FIRST — Roman Hindi words
    hindi_overlap = len(words & _HINDI_ROMAN_WORDS)
    if hindi_overlap >= 2:
        return "hinglish"

    # langdetect for all other languages
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        code = detect(text_str)
        lang_map = {
            "hi":"hindi","ur":"urdu","es":"spanish","fr":"french",
            "de":"german","ar":"arabic","ru":"russian","en":"english",
            "zh-cn":"chinese","zh-tw":"chinese","pt":"portuguese",
            "it":"italian","ja":"japanese","ko":"korean"
        }
        detected = lang_map.get(code, "english")
        # Edge case: langdetect says English but has Hindi words
        if detected == "english" and hindi_overlap >= 1:
            return "hinglish"
        return detected
    except Exception:
        return "english"


@st.cache_resource
def load_sentiment_model():
    """
    Load XLM-RoBERTa multilingual sentiment model.
    Pre-trained on 100+ languages — no translation needed.
    """
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None, truncation=True, max_length=512
        )
    except Exception as e:
        logger.warning(f"Sentiment model load failed: {e}")
        return None

# Alias for compatibility
load_multilingual_model = load_sentiment_model


def _xlm_predict(text, model):
    """Run XLM-RoBERTa directly on text."""
    try:
        results = model(str(text)[:512])
        scores = results[0] if isinstance(results[0], list) else results
        best = max(scores, key=lambda x: x["score"])
        label = best["label"].lower()
        score = round(best["score"], 3)
        if label in ["positive", "pos", "label_2", "2"]:
            return "Positive", score
        elif label in ["negative", "neg", "label_0", "0"]:
            return "Negative", score
        else:
            return "Neutral", score
    except Exception:
        return None, None


def _vader_predict(text):
    """VADER sentiment — used as supporting signal."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        c = SentimentIntensityAnalyzer().polarity_scores(str(text))["compound"]
        if c >= 0.05:    return "Positive", round(abs(c), 3)
        elif c <= -0.05: return "Negative", round(abs(c), 3)
        else:            return "Neutral",  round(1 - abs(c), 3)
    except Exception:
        return "Neutral", 0.5


def _hinglish_lexicon_predict(text):
    """
    Hinglish-specific lexicon scoring.
    Original contribution: Roman-Hindi word sentiment matching
    with intensifier boosting.
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    intensifier_boost = 1.0 + (len(words & _HINGLISH_INTENSIFIERS) * 0.2)
    pos = len(words & _HINGLISH_POSITIVE) * intensifier_boost
    neg = len(words & _HINGLISH_NEGATIVE) * intensifier_boost
    if pos > neg:    return "Positive", min(0.5 + pos * 0.12, 0.95)
    elif neg > pos:  return "Negative", min(0.5 + neg * 0.12, 0.95)
    else:            return None, None


def multilingual_sentiment(text, model=None):
    """
    MASTER SENTIMENT FUNCTION
    ─────────────────────────
    Pipeline:
      English/Hindi/Urdu/Spanish/etc → XLM-RoBERTa (direct, no translation)
      Hinglish → Lexicon + VADER + XLM majority voting (original contribution)

    Fallback chain: XLM → VADER → Neutral
    """
    text = str(text).strip()
    if not text or len(text) < 2:
        return "Neutral", 0.5

    lang = detect_text_language(text)

    # ── Hinglish: 3-signal majority voting ────────────────────────────────────
    if lang == "hinglish":
        lex_sent, lex_score   = _hinglish_lexicon_predict(text)
        vader_sent, vader_sc  = _vader_predict(text)
        xlm_sent,  xlm_sc     = _xlm_predict(text, model) if model else (None, None)

        votes = [s for s in [lex_sent, vader_sent, xlm_sent] if s]
        if votes:
            from collections import Counter
            winner = Counter(votes).most_common(1)[0][0]
            # Score from most confident signal
            score = xlm_sc or lex_score or vader_sc or 0.6
            return winner, round(score, 3)
        return _vader_predict(text)

    # ── All other languages: XLM-RoBERTa direct ───────────────────────────────
    else:
        if model is not None:
            sent, score = _xlm_predict(text, model)
            if sent:
                return sent, score
        # Fallback to VADER
        return _vader_predict(text)


def apply_multilingual_analysis(df, show_progress=True, include_lang_col=True):
    """
    Apply multilingual sentiment analysis to entire DataFrame.
    Uses XLM-RoBERTa for all languages + special Hinglish handling.
    """
    model = load_sentiment_model()
    df = df.copy()
    sentiments, scores, languages = [], [], []
    total = len(df)

    lang_flags = {
        "english":"🇬🇧","hindi":"🇮🇳","hinglish":"🤝","urdu":"🇵🇰",
        "spanish":"🇪🇸","french":"🇫🇷","german":"🇩🇪","arabic":"🇸🇦",
        "russian":"🇷🇺","chinese":"🇨🇳","japanese":"🇯🇵","korean":"🇰🇷"
    }

    progress_bar = None

    for i, text in enumerate(df["message"].fillna("").astype(str).tolist()):
        lang = detect_text_language(text)
        languages.append(lang)
        sent, score = multilingual_sentiment(text, model)
        sentiments.append(sent)
        scores.append(score)




    df["sentiment"]       = sentiments
    df["sentiment_score"] = scores
    if include_lang_col:
        df["detected_language"] = languages
    return df

try:
    from src.advanced_sentiment import apply_advanced_sentiment, compare_sentiments
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError as e:
    ADVANCED_SENTIMENT_AVAILABLE = False
    logger.warning(f"Advanced sentiment import failed: {e}")

try:
    from src.multimodal_sentiment import apply_multimodal_sentiment, detect_language, get_available_models
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    MULTIMODAL_AVAILABLE = False
    logger.warning(f"Multimodal import failed: {e}")

SENTIMENT_CONFIDENCE_THRESHOLD = 0.3

# ─── THEME STATE ───
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "include_emotion" not in st.session_state:
    st.session_state.include_emotion = True
if "analysis_method_index" not in st.session_state:
    st.session_state.analysis_method_index = 0

def get_theme_css(theme):
    if theme == "dark":
        return """
        --bg:           #0d1117;
        --bg2:          #161b22;
        --card-bg:      rgba(22,27,34,0.90);
        --card-border:  rgba(88,166,255,0.18);
        --text:         #e6edf3;
        --text-muted:   #8b949e;
        --accent1:      #58a6ff;
        --accent2:      #bc8cff;
        --accent3:      #3fb950;
        --accent4:      #f78166;
        --glow1:        rgba(88,166,255,0.20);
        --glow2:        rgba(188,140,255,0.15);
        --wc1:          rgba(88,166,255,0.07);
        --wc2:          rgba(188,140,255,0.05);
        --wc3:          rgba(63,185,80,0.05);
        --sidebar-bg:   #0d1117;
        --sidebar-text: #c9d1d9;
        --divider:      rgba(255,255,255,0.08);
        --shadow:       0 8px 32px rgba(0,0,0,0.4);
        --header-grad:  linear-gradient(135deg,#1a2744 0%,#1e1b4b 60%,#0d1117 100%);
        --tab-active:   linear-gradient(135deg,#58a6ff,#bc8cff);
        --positive:     #3fb950;
        --negative:     #f78166;
        --neutral:      #58a6ff;
        """
    else:
        return """
        --bg:           #f4f0ff;
        --bg2:          #ffffff;
        --card-bg:      rgba(255,255,255,0.85);
        --card-border:  rgba(99,102,241,0.20);
        --text:         #1a1a2e;
        --text-muted:   #6b7280;
        --accent1:      #6366f1;
        --accent2:      #8b5cf6;
        --accent3:      #10b981;
        --accent4:      #f59e0b;
        --glow1:        rgba(99,102,241,0.15);
        --glow2:        rgba(139,92,246,0.12);
        --wc1:          rgba(167,139,250,0.12);
        --wc2:          rgba(196,181,253,0.15);
        --wc3:          rgba(167,243,208,0.18);
        --sidebar-bg:   #1a1a2e;
        --sidebar-text: #e2e8f0;
        --divider:      rgba(0,0,0,0.07);
        --shadow:       0 8px 32px rgba(99,102,241,0.10);
        --header-grad:  linear-gradient(135deg,#4f46e5 0%,#7c3aed 60%,#6366f1 100%);
        --tab-active:   linear-gradient(135deg,#6366f1,#8b5cf6);
        --positive:     #10b981;
        --negative:     #ef4444;
        --neutral:      #6366f1;
        """

theme_css = get_theme_css(st.session_state.theme)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
    {theme_css}
    --transition: 0.35s cubic-bezier(.4,0,.2,1);
    --radius: 16px;
}}

* {{ font-family: 'DM Sans', sans-serif !important; }}

/* ─── APP BACKGROUND ─── */
.stApp {{
    background: var(--bg) !important;
    transition: background var(--transition);
}}
.main .block-container {{
    background: transparent !important;
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
    position: relative;
}}

/* Watercolor animated background */
.main .block-container::before {{
    content: '';
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(ellipse 55% 35% at 8%  15%,  var(--wc1) 0%, transparent 60%),
        radial-gradient(ellipse 45% 45% at 92% 8%,   var(--wc2) 0%, transparent 60%),
        radial-gradient(ellipse 65% 35% at 55% 92%,  var(--wc3) 0%, transparent 55%),
        radial-gradient(ellipse 40% 55% at 78% 58%,  var(--wc1) 0%, transparent 50%),
        radial-gradient(ellipse 50% 40% at 20% 75%,  var(--wc2) 0%, transparent 50%);
    animation: wc-breathe 10s ease-in-out infinite alternate;
}}
@keyframes wc-breathe {{
    0%   {{ opacity:1; transform:scale(1); }}
    100% {{ opacity:0.75; transform:scale(1.04); }}
}}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--divider) !important;
    position: relative; overflow: hidden;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
    min-width: 280px !important;
    left: 0 !important;
}}
section[data-testid="stSidebar"]::before {{
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 80% 25% at 50% 0%,   rgba(88,166,255,0.13) 0%, transparent 70%),
        radial-gradient(ellipse 60% 35% at 80% 100%,  rgba(188,140,255,0.10) 0%, transparent 60%);
}}
/* Keep sidebar always dark regardless of theme */
section[data-testid="stSidebar"] {{
    background: #1a1a2e !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
    min-width: 280px !important;
}}
/* Force ALL sidebar text to be white */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] small {{
    color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] .stCheckbox label {{
    color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #e2e8f0 !important;
}}

/* Radio buttons — beautiful */
section[data-testid="stSidebar"] .stRadio > div {{
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    background: rgba(255,255,255,0.03) !important;
    cursor: pointer !important;
    transition: all 0.25s !important;
    margin: 0 !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(88,166,255,0.08) !important;
    border-color: rgba(88,166,255,0.25) !important;
    transform: translateX(3px);
}}
section[data-testid="stSidebar"] input[type="radio"] {{
    appearance: auto !important;
    -webkit-appearance: radio !important;
    display: inline-block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 15px !important;
    height: 15px !important;
    margin-right: 10px !important;
    cursor: pointer !important;
    accent-color: var(--accent1) !important;
    flex-shrink: 0 !important;
}}

/* File uploader */
[data-testid="stFileUploader"] {{
    background: rgba(88,166,255,0.04) !important;
    border: 2px dashed rgba(88,166,255,0.30) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    transition: border-color 0.3s !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: var(--accent1) !important;
    background: rgba(88,166,255,0.07) !important;
}}

/* ─── HEADER BANNER ─── */
.main-header {{
    background: var(--header-grad);
    padding: 30px 40px;
    border-radius: 20px;
    margin-bottom: 28px;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow), 0 0 60px var(--glow1);
    display: flex; align-items: center; justify-content: space-between;
}}
.main-header::before {{
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 50% 80% at 100% 50%, rgba(188,140,255,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 0% 0%,   rgba(88,166,255,0.18) 0%, transparent 50%);
}}
.main-header::after {{
    content: '✦  ✦  ✦';
    position: absolute; right: 40px; top: 14px;
    font-size: 0.55rem; color: rgba(255,255,255,0.18);
    letter-spacing: 10px; pointer-events: none;
}}
.main-header h1 {{
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important; font-weight: 900 !important;
    color: white !important;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3) !important;
    margin: 0 0 6px 0 !important;
    position: relative; z-index: 1;
}}
.main-header p {{
    color: rgba(255,255,255,0.72) !important;
    font-size: 0.95rem !important; font-weight: 300 !important;
    margin: 0 !important; position: relative; z-index: 1;
}}
.main-header .badge {{
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.22);
    color: white !important; padding: 8px 22px;
    border-radius: 30px; font-size: 0.8rem !important;
    font-weight: 600 !important; white-space: nowrap;
    position: relative; z-index: 1;
}}

/* ─── KPI METRIC CARDS ─── */
.metric-card {{
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    padding: 22px 24px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: var(--shadow) !important;
    position: relative; overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s !important;
}}
.metric-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
}}
.metric-card::after {{
    content: '';
    position: absolute; bottom: -25px; right: -25px;
    width: 90px; height: 90px; border-radius: 50%;
    background: radial-gradient(circle, var(--wc1) 0%, transparent 70%);
    pointer-events: none;
}}
.metric-card:hover {{
    transform: translateY(-5px) !important;
    box-shadow: 0 16px 48px var(--glow1) !important;
}}
.kpi-number {{
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1 !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}}
.kpi-label {{
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}}
.metric-card p, .metric-card div, .metric-card span {{
    color: var(--text) !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--card-bg) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    box-shadow: var(--shadow) !important;
    gap: 4px !important;
    border: 1px solid var(--card-border) !important;
    backdrop-filter: blur(20px) !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    padding: 9px 18px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    transition: all 0.25s !important;
}}
.stTabs [aria-selected="true"] {{
    background: var(--tab-active) !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px var(--glow1) !important;
}}

/* ─── CHART CARDS ─── */
.chart-container {{
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    padding: 22px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 18px !important;
    position: relative; overflow: hidden;
}}
.chart-container::before {{
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(ellipse 60% 40% at 90% 10%, var(--wc2) 0%, transparent 60%);
}}

/* ─── BUTTONS ─── */
.stButton > button {{
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 16px var(--glow1) !important;
    transition: all 0.25s !important;
    letter-spacing: 0.3px !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--glow1) !important;
}}

/* ─── DOWNLOAD BUTTON FIX ─── */
.stDownloadButton > button {{
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 16px var(--glow1) !important;
    transition: all 0.25s !important;
}}
.stDownloadButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--glow1) !important;
    opacity: 0.9 !important;
}}
.stDownloadButton > button p {{
    color: white !important;
}}
.stTextInput input, .stTextArea textarea {{
    background: var(--card-bg) !important;
    border: 1.5px solid var(--card-border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
    transition: border 0.25s !important;
    padding: 10px 14px !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 3px var(--glow1) !important;
}}

/* ─── DATAFRAME ─── */
.stDataFrame {{
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
    border: 1px solid var(--card-border) !important;
}}

/* ─── SELECT / MULTISELECT ─── */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: var(--card-bg) !important;
    border-color: var(--card-border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}}

/* ─── ALERTS ─── */
.stSuccess {{
    background: rgba(63,185,80,0.10) !important;
    border-left: 4px solid var(--positive) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}}
.stInfo {{
    background: rgba(88,166,255,0.08) !important;
    border-left: 4px solid var(--accent1) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}}
.stWarning {{
    background: rgba(247,129,102,0.08) !important;
    border-left: 4px solid var(--accent4) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}}

/* ─── SECTION HEADERS ─── */
h1, h2, h3 {{
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}}

/* ─── GLOBAL TEXT VISIBILITY FIX ─── */
.stApp p, .stApp span, .stApp div, .stApp label {{
    color: var(--text) !important;
}}
.stApp .stMarkdown p {{
    color: var(--text) !important;
}}
/* Streamlit native metric */
[data-testid="stMetricValue"] {{
    color: var(--text) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: var(--text-muted) !important;
}}
/* Summary text */
.stApp .element-container p {{
    color: var(--text) !important;
}}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: var(--accent1);
    border-radius: 3px; opacity: 0.6;
}}

/* ─── HIDE STREAMLIT CHROME ─── */
.stDeployButton {{display:none !important;}}
header[data-testid="stHeader"] {{display:none !important; height:0 !important;}}
#MainMenu {{display:none !important;}}
footer {{display:none !important;}}
[data-testid="stToolbar"] {{display:none !important;}}
div[data-testid="stDecoration"] {{display:none !important;}}
.block-container {{padding-top: 0.5rem !important;}}

/* ─── FORCE SIDEBAR ALWAYS VISIBLE ─── */
section[data-testid="stSidebar"] {{
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
    min-width: 280px !important;
    left: 0 !important;
}}
[data-testid="collapsedControl"] {{
    display: none !important;
}}

/* ─── THEME TOGGLE BUTTON ─── */
.theme-btn {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 18px; border-radius: 20px;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--divider);
    cursor: pointer; font-size: 0.82rem;
    color: var(--sidebar-text);
    transition: all 0.3s;
    width: 100%;
    justify-content: center;
    font-weight: 500;
}}
.theme-btn:hover {{
    background: rgba(88,166,255,0.12);
    border-color: var(--accent1);
}}

/* ─── SLIDER ─── */
.stSlider > div > div > div {{
    background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
}}

/* ─── PROGRESS BAR ─── */
.stProgress > div > div {{
    background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
    border-radius: 4px !important;
}}

/* section divider */
.sec-divider {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--card-border), transparent);
    margin: 20px 0;
}}

/* sidebar section label */
.sb-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent1);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.sb-label::after {{
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(to right, var(--accent1), transparent);
    opacity: 0.35;
}}
</style>
""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ───
@st.cache_resource
def load_advanced_sentiment_analyzer(sentiment_model=None, emotion_model=None):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"device": device}
        if sentiment_model:
            kwargs["sentiment_model"] = sentiment_model
        if emotion_model:
            kwargs["emotion_model"] = emotion_model
        analyzer = AdvancedSentimentAnalyzer(**kwargs)
        return analyzer
    except Exception as e:
        st.error(f"Failed to load advanced sentiment model: {e}")
        return None


def predict_sentiment(text, analyzer):
    if not analyzer:
        return "Neutral", 0.0
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return "Neutral", 1.0
    result = analyzer.analyze_sentiment(cleaned_text)
    raw_label = result["label"].lower()
    score = float(result["score"])
    label_map = {"negative": "negative", "neutral": "neutral", "positive": "positive",
                 "label_0": "negative", "label_1": "neutral", "label_2": "positive",
                 "label_3": "positive", "label_4": "positive"}
    label = label_map.get(raw_label, raw_label)
    if "star" in label:
        try:
            stars = int(label.split()[0])
            label = "negative" if stars <= 2 else ("neutral" if stars == 3 else "positive")
        except:
            label = "neutral"
    if score < SENTIMENT_CONFIDENCE_THRESHOLD:
        return "neutral", score
    return label, score


def predict_emotion(text, analyzer):
    if not analyzer or not analyzer.emotion_pipeline:
        return "neutral", 0.0
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return "neutral", 1.0
    result = analyzer.analyze_emotion(cleaned_text)
    label = result["emotion"]
    score = float(result["score"])
    if "star" in label:
        try:
            stars = int(label.split()[0])
            label = "negative" if stars <= 2 else ("neutral" if stars == 3 else "positive")
        except:
            label = "neutral"
    return label, score


def analyze_single_text_transformer(text, analyzer, confidence_threshold=SENTIMENT_CONFIDENCE_THRESHOLD):
    sentiment, score = predict_sentiment(text, analyzer)
    emotion_result = None
    if st.session_state.get("include_emotion", False) and analyzer and analyzer.emotion_pipeline:
        emotion, e_score = predict_emotion(text, analyzer)
        emotion_result = {"emotion": emotion, "score": e_score}
    return sentiment, score, emotion_result


@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="t5-small")
    except Exception as e:
        logger.error(f"Failed to load summarizer: {e}")
        return None


def generate_text_summary(df):
    summarizer = load_summarizer()
    if not summarizer:
        return "Summarizer model not available."
    sample_df = df.tail(300)
    text = " ".join(sample_df[~sample_df["message"].str.contains("<Media omitted>", na=False)]["message"].astype(str).tolist())
    if not text.strip() or len(text.split()) < 50:
        return "Not enough text to generate a meaningful summary."
    try:
        with st.spinner("Generating chat summary..."):
            result = summarizer("summarize: " + text, max_length=150, min_length=40, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        return f"Could not generate summary: {str(e)}"


def clean_text_for_wordcloud(df_row):
    if df_row.get("is_system_message", False) or df_row.get("is_emoji_only", False):
        return ""
    text = df_row.get("message_raw", "")
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"<Media omitted>|This message was deleted|https?://\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\s]|\d+", "", text)
    words = [w for w in text.lower().split() if len(w) >= 3 and w not in stop_words]
    return " ".join(words)


def detect_emotion(text):
    try:
        em = NRCLex(text)
        return em.top_emotions[0][0] if em.top_emotions else "neutral"
    except:
        return "neutral"


def get_emoji_sentiment(text):
    pos = ["😊", "😂", "😍", "👍", "🎉", "❤️"]
    neg = ["😠", "😢", "👎", "💔"]
    emojis = [c["emoji"] for c in emoji.emoji_list(text)]
    for e in emojis:
        if e in pos:
            return "positive"
        if e in neg:
            return "negative"
    return None


def live_prediction(text):
    if not str(text).strip():
        return "Neutral", 0.5, "Neutral"

    # Detect language
    lang = detect_text_language(text)

    # Use XLM-RoBERTa multilingual model
    model = load_multilingual_model()
    sent, score = multilingual_sentiment(text, model)

    # Emotion detection
    emotion_label = "Neutral"
    if st.session_state.get("include_emotion", False):
        try:
            a = load_advanced_sentiment_analyzer(
                emotion_model="j-hartmann/emotion-english-distilroberta-base")
            if a:
                em, _ = predict_emotion(text, a)
                emotion_label = em.capitalize()
        except Exception:
            pass

    # Emoji override
    es = get_emoji_sentiment(text)
    if es:
        sent = es.capitalize()
        score = 0.9

    return sent, score, emotion_label


def calculate_performance_metrics(y_true, y_pred):
    labels = sorted(list(set(y_true) | set(y_pred)))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return accuracy, precision, recall, f1, cm, labels


# ─── CHART THEME ───
SENTIMENT_COLORS = {"positive": "#3fb950", "negative": "#f78166", "neutral": "#58a6ff",
                    "Positive": "#3fb950", "Negative": "#f78166", "Neutral": "#58a6ff"}
EMOTION_COLORS = {"joy": "#fbbf24", "sadness": "#3b82f6", "anger": "#ef4444", "fear": "#8b5cf6",
                  "surprise": "#ec4899", "love": "#f43f5e", "neutral": "#6b7280",
                  "disgust": "#84cc16", "positive": "#3fb950", "negative": "#f78166"}

def apply_chart_theme(fig, title):
    is_dark = st.session_state.theme == "dark"
    bg = "#161b22" if is_dark else "#ffffff"
    text_color = "#e6edf3" if is_dark else "#1a1a2e"
    grid_color = "#21262d" if is_dark else "#f0f0f8"
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        title={"text": title, "x": 0.05,
               "font": {"size": 17, "family": "Playfair Display", "color": text_color}},
        paper_bgcolor=bg, plot_bgcolor=bg,
        font={"family": "DM Sans", "color": text_color},
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        margin=dict(l=40, r=40, t=55, b=40)
    )
    return fig


def ui_metric_card(label, value, icon="📊", color="accent1"):
    is_dark = st.session_state.get("theme", "dark") == "dark"
    text_color = "#e6edf3" if is_dark else "#1a1a2e"
    muted_color = "#8b949e" if is_dark else "#6b7280"
    card_bg = "rgba(22,27,34,0.90)" if is_dark else "rgba(255,255,255,0.85)"
    border_color = "rgba(88,166,255,0.18)" if is_dark else "rgba(99,102,241,0.20)"
    st.markdown(f"""
        <div style="background:{card_bg};border:1px solid {border_color};
                    border-radius:16px;padding:22px 24px;
                    backdrop-filter:blur(20px);
                    box-shadow:0 8px 32px rgba(0,0,0,0.3);
                    position:relative;overflow:hidden;margin-bottom:8px;
                    border-top:3px solid;
                    border-image:linear-gradient(90deg,#58a6ff,#bc8cff) 1;">
            <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:1.2px;color:{muted_color};margin-bottom:8px;">
                {icon} {label}
            </div>
            <div style="font-family:'Playfair Display',serif;font-size:2.4rem;
                        font-weight:700;color:{text_color};line-height:1;">
                {value}
            </div>
        </div>
    """, unsafe_allow_html=True)


def ui_divider():
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)


def ui_header_banner(subtitle=""):
    th = "🌙 Dark" if st.session_state.theme == "dark" else "☀️ Light"
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1>💬 WhatsApp Sentiment Analyzer</h1>
            <p>AI-powered sentiment & emotion analysis · {subtitle}</p>
        </div>
        <div class="badge">B.Tech Final Year Project</div>
    </div>
    """, unsafe_allow_html=True)


def create_message_heatmap(df):
    if df.empty or "datetime" not in df.columns:
        return go.Figure()
    df_copy = df.copy()
    df_copy["day_of_week"] = df_copy["datetime"].dt.day_name()
    df_copy["hour"] = df_copy["datetime"].dt.hour
    heatmap_data = df_copy.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index], fill_value=0)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h:02d}:00" for h in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale="Viridis",
        showscale=True
    ))
    return apply_chart_theme(fig, "📅 Message Activity Heatmap")


def create_sentiment_pie_chart(df, sentiment_col):
    if df.empty or sentiment_col not in df.columns:
        return go.Figure()
    counts = df[sentiment_col].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.45,
        marker=dict(colors=[SENTIMENT_COLORS.get(s, "#6b7280") for s in counts.index],
                    line=dict(color="rgba(0,0,0,0.2)", width=2)),
        textinfo="label+percent",
        textfont=dict(family="DM Sans", size=13)
    )])
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return apply_chart_theme(fig, "😊 Sentiment Distribution")


def create_sentiment_timeline(df, sentiment_col):
    if df.empty or sentiment_col not in df.columns or "datetime" not in df.columns:
        return go.Figure()
    df_copy = df.copy()
    df_copy["date"] = df_copy["datetime"].dt.floor("D")
    data = df_copy.groupby(["date", sentiment_col]).size().unstack(fill_value=0)
    fig = go.Figure()
    for s in data.columns:
        color = SENTIMENT_COLORS.get(s, "#6b7280")
        # Convert hex to rgba for fillcolor
        def hex_to_rgba(hex_color, alpha=0.08):
            hex_color = hex_color.lstrip("#")
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        fill_color = hex_to_rgba(color) if color.startswith("#") else color
        fig.add_trace(go.Scatter(
            x=data.index, y=data[s], name=s.capitalize(),
            mode="lines+markers",
            line=dict(width=3, color=color, shape="spline"),
            marker=dict(size=8, color=color,
                        line=dict(color="white", width=1.5)),
            fill="tozeroy",
            fillcolor=fill_color
        ))
    fig.update_layout(hovermode="x unified", xaxis=dict(type="date", tickformat="%Y-%m-%d"))
    return apply_chart_theme(fig, "📈 Sentiment Trends Over Time")


def create_sentiment_comparison_chart(df, sentiment_col):
    if df.empty or sentiment_col not in df.columns or "author" not in df.columns:
        return go.Figure()
    user_summary = df.groupby(["author", sentiment_col]).size().unstack(fill_value=0)
    fig = go.Figure()
    for s in user_summary.columns:
        fig.add_trace(go.Bar(
            name=s.capitalize(), x=user_summary.index, y=user_summary[s],
            marker=dict(color=SENTIMENT_COLORS.get(s, "#6b7280"),
                        line=dict(color="rgba(0,0,0,0.1)", width=1)),
        ))
    fig.update_layout(barmode="group")
    return apply_chart_theme(fig, "👥 Sentiment by User")


def create_emotion_sunburst(df, emotion_col):
    if df.empty or emotion_col not in df.columns:
        return go.Figure()
    counts = df[emotion_col].value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    counts["parent"] = "All Emotions"
    root = pd.DataFrame([{"emotion": "All Emotions", "parent": "", "count": counts["count"].sum()}])
    emotion_data = pd.concat([root, counts], ignore_index=True)
    fig = go.Figure(go.Sunburst(
        labels=emotion_data["emotion"], parents=emotion_data["parent"],
        values=emotion_data["count"], branchvalues="total",
        marker=dict(colors=[EMOTION_COLORS.get(e.lower(), "#6b7280") for e in emotion_data["emotion"]]),
        textfont=dict(family="DM Sans")
    ))
    return apply_chart_theme(fig, "🌈 Emotion Distribution")


def create_emotion_timeline(df, emotion_col):
    if df.empty or emotion_col not in df.columns or "datetime" not in df.columns:
        return go.Figure()
    df_copy = df.copy()
    df_copy["date"] = df_copy["datetime"].dt.floor("D")
    data = df_copy.groupby(["date", emotion_col]).size().unstack(fill_value=0)
    fig = go.Figure()
    for em in data.columns:
        color = EMOTION_COLORS.get(em.lower(), "#6b7280")
        fig.add_trace(go.Scatter(
            x=data.index, y=data[em], name=em.capitalize(),
            mode="lines+markers", line=dict(width=2, color=color, shape="spline"),
            stackgroup="one"
        ))
    return apply_chart_theme(fig, "📊 Emotion Trends Over Time")


def create_emotion_by_user_chart(df, emotion_col):
    if df.empty or emotion_col not in df.columns or "author" not in df.columns:
        return go.Figure()
    df_c = df.copy()
    if df_c[emotion_col].apply(lambda x: isinstance(x, list)).any():
        df_c[emotion_col] = df_c[emotion_col].apply(lambda x: str(x) if isinstance(x, list) else x)
    user_summary = df_c.groupby(["author", emotion_col]).size().unstack(fill_value=0)
    fig = go.Figure()
    for em in user_summary.columns:
        fig.add_trace(go.Bar(
            name=em.capitalize(), x=user_summary.index, y=user_summary[em],
            marker=dict(color=EMOTION_COLORS.get(em.lower(), "#6b7280"))
        ))
    return apply_chart_theme(fig, "👤 Emotion by User")


def create_word_frequency_chart(df, top_n=20):
    if df.empty or "message" not in df.columns:
        return go.Figure()
    text = " ".join(df["message"].astype(str).tolist())
    words = [w.lower() for w in text.split() if len(w) > 3]
    wc = Counter(words).most_common(top_n)
    is_dark = st.session_state.theme == "dark"
    fig = go.Figure(data=[go.Bar(
        x=[c for _, c in wc], y=[w for w, _ in wc],
        orientation="h",
        marker=dict(
            color=list(range(len(wc))),
            colorscale="Viridis",
            line=dict(color="rgba(0,0,0,0.1)", width=1)
        )
    )])
    return apply_chart_theme(fig, f"☁️ Top {top_n} Words")


def create_user_activity_gauge(df, user):
    if df.empty or "author" not in df.columns:
        return go.Figure()
    pct = max(0, (len(df[df["author"] == user]) / len(df) * 100) if len(df) > 0 else 0)
    is_dark = st.session_state.theme == "dark"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=pct,
        title={"text": f"{user}'s Share", "font": {"size": 16, "family": "Playfair Display"}},
        delta={"reference": 50, "increasing": {"color": "#3fb950"}},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "#58a6ff"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 33], "color": "rgba(88,166,255,0.05)"},
                {"range": [33, 66], "color": "rgba(88,166,255,0.10)"},
                {"range": [66, 100], "color": "rgba(88,166,255,0.15)"}
            ],
            "threshold": {"line": {"color": "#f78166", "width": 4}, "thickness": 0.75, "value": 90}
        }
    ))
    return apply_chart_theme(fig, "")


def create_daily_average_compound_chart(df):
    if df.empty or "vader_compound" not in df.columns or "datetime" not in df.columns:
        return go.Figure()
    df_c = df.copy()
    df_c["date"] = df_c["datetime"].dt.floor("D")
    daily = df_c.groupby("date")["vader_compound"].mean().reset_index()
    fig = go.Figure(data=[go.Scatter(
        x=daily["date"], y=daily["vader_compound"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=3, shape="spline"),
        marker=dict(size=8, color="#58a6ff", line=dict(color="white", width=1.5)),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
    )])
    return apply_chart_theme(fig, "📉 VADER Compound Trend")


def create_hourly_average_compound_chart(df):
    if df.empty or "vader_compound" not in df.columns or "datetime" not in df.columns:
        return go.Figure()
    df_c = df.copy()
    df_c["hour"] = df_c["datetime"].dt.hour
    hourly = df_c.groupby("hour")["vader_compound"].mean().reset_index()
    hourly["hour_label"] = hourly["hour"].apply(lambda x: f"{x:02d}:00")
    fig = go.Figure(data=[go.Bar(
        x=hourly["hour_label"], y=hourly["vader_compound"],
        marker=dict(color=hourly["vader_compound"], colorscale="RdYlGn",
                    line=dict(color="rgba(0,0,0,0.1)", width=1))
    )])
    return apply_chart_theme(fig, "⏰ Avg Sentiment by Hour")


def create_confidence_score_chart(df):
    if df.empty or "sentiment_score" not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x="sentiment_score", nbins=20,
                       color_discrete_sequence=["#58a6ff"])
    return apply_chart_theme(fig, "🎯 Confidence Score Distribution")


def create_sentiment_emotion_comparison(df):
    ec = "transformer_emotion" if "transformer_emotion" in df.columns else ("emotion" if "emotion" in df.columns else None)
    if not ec or "sentiment" not in df.columns:
        return go.Figure()
    comparison = pd.crosstab(df["sentiment"], df[ec])
    fig = px.imshow(comparison, text_auto=True, aspect="auto",
                    color_continuous_scale="Viridis")
    return apply_chart_theme(fig, "🔄 Sentiment vs Emotion")


def generate_pdf_report(df, sentiment_col):
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "WhatsApp Sentiment Analysis Report", 1, 0, "C")
            self.ln(20)
        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "WhatsApp Sentiment Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Total Messages: {len(df)}\nTotal Users: {df['user'].nunique()}\nDate Range: {pd.to_datetime(df['datetime']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(df['datetime']).max().strftime('%Y-%m-%d')}")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Sentiment Distribution", ln=True)
    pdf.set_font("Arial", size=12)
    for s, c in df[sentiment_col].value_counts().items():
        pdf.cell(0, 10, f"{s}: {c}", ln=True)
    pdf.ln(10)
    score_col = "sentiment_score" if "sentiment_score" in df.columns else "vader_compound"
    for label, ascending in [("Most Positive", False), ("Most Negative", True)]:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Top 3 {label} Messages", ln=True)
        pdf.set_font("Arial", size=10)
        polarity = "positive" if not ascending else "negative"
        msgs = df[df[sentiment_col].str.lower() == polarity].sort_values(score_col, ascending=ascending).head(3)
        for _, row in msgs.iterrows():
            msg = str(row["message"]).encode("latin-1", "replace").decode("latin-1").strip()
            pdf.multi_cell(0, 5, f"- {msg}", 0, "L")
            pdf.ln(2)
        pdf.ln(10)
    return bytes(pdf.output())


def render_top_emotional_words(df, emotion_col):
    EMOTION_ICONS = {"joy":"😊","sadness":"😢","anger":"😠","fear":"😨",
                     "surprise":"😲","love":"❤️","neutral":"😐","disgust":"🤢",
                     "positive":"✅","negative":"❌"}
    EMOTION_GRADIENTS = {
        "joy":     ("rgba(251,191,36,0.12)",  "#fbbf24"),
        "sadness": ("rgba(59,130,246,0.12)",   "#3b82f6"),
        "anger":   ("rgba(239,68,68,0.12)",    "#ef4444"),
        "fear":    ("rgba(139,92,246,0.12)",   "#8b5cf6"),
        "surprise":("rgba(236,72,153,0.12)",   "#ec4899"),
        "love":    ("rgba(244,63,94,0.12)",    "#f43f5e"),
        "neutral": ("rgba(107,114,128,0.12)",  "#6b7280"),
        "disgust": ("rgba(132,204,22,0.12)",   "#84cc16"),
        "positive":("rgba(63,185,80,0.12)",    "#3fb950"),
        "negative":("rgba(247,129,102,0.12)",  "#f78166"),
    }
    is_dark = st.session_state.get("theme","dark") == "dark"
    text_color = "#e6edf3" if is_dark else "#1a1a2e"
    muted_color = "#8b949e" if is_dark else "#6b7280"
    card_bg = "rgba(22,27,34,0.90)" if is_dark else "rgba(255,255,255,0.90)"

    if emotion_col and not df[emotion_col].isna().all():
        emotion_words = {}
        for el in df[emotion_col].unique():
            if pd.notna(el):
                emotion_words[el] = " ".join(df[df[emotion_col] == el]["message"].astype(str).tolist())

        items = [(el, text) for el, text in emotion_words.items() if text.strip()]
        for row_start in range(0, len(items), 3):
            row_items = items[row_start:row_start+3]
            cols = st.columns(len(row_items))
            for col, (el, text) in zip(cols, row_items):
                words = [w.lower() for w in text.split() if len(w) > 3]
                wc = Counter(words).most_common(8)
                if not wc:
                    continue
                key = el.lower()
                bg, accent = EMOTION_GRADIENTS.get(key, ("rgba(88,166,255,0.10)", "#58a6ff"))
                icon = EMOTION_ICONS.get(key, "💬")
                rows_html = ""
                for rank, (word, freq) in enumerate(wc):
                    bar_w = int(freq / wc[0][1] * 100)
                    rows_html += f"""
                    <div style="display:flex;align-items:center;gap:10px;
                                padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="font-size:0.72rem;color:{muted_color};
                                     width:16px;text-align:right;">{rank+1}</span>
                        <span style="font-size:0.85rem;color:{text_color};
                                     font-weight:500;flex:1;">{word}</span>
                        <div style="width:60px;height:5px;border-radius:3px;
                                    background:rgba(255,255,255,0.08);overflow:hidden;">
                            <div style="width:{bar_w}%;height:100%;
                                        background:{accent};border-radius:3px;"></div>
                        </div>
                        <span style="font-size:0.75rem;color:{accent};
                                     font-weight:700;width:20px;text-align:right;">{freq}</span>
                    </div>"""
                with col:
                    st.markdown(f"""
                    <div style="background:{card_bg};border:1px solid {accent}30;
                                border-radius:16px;padding:18px;margin-bottom:16px;
                                border-top:3px solid {accent};
                                box-shadow:0 4px 20px {accent}15;">
                        <div style="font-size:1.1rem;font-weight:700;color:{text_color};
                                    margin-bottom:14px;font-family:'Playfair Display',serif;">
                            {icon} {el.capitalize()}
                        </div>
                        {rows_html}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No emotion data available.")
        st.info("Enable 'Include Emotion Detection' in sidebar.")


def render_sentiment_section(df, sentiment_col, key_prefix="default"):
    st.plotly_chart(create_sentiment_timeline(df, sentiment_col),
                    use_container_width=True, key=f"sent_timeline_{key_prefix}")
    ui_divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_sentiment_pie_chart(df, sentiment_col),
                        use_container_width=True, key=f"sent_pie_{key_prefix}")
    with c2:
        st.plotly_chart(create_sentiment_comparison_chart(df, sentiment_col),
                        use_container_width=True, key=f"sent_comp_{key_prefix}")


def render_emotion_section(df, key_prefix="default"):
    ec = ("transformer_emotion" if "transformer_emotion" in df.columns
          else ("mm_emotion" if "mm_emotion" in df.columns
          else ("emotion" if "emotion" in df.columns else None)))
    if ec and not df[ec].isna().all():
        unique_labels = [str(l) for l in df[ec].dropna().unique()]
        st.info(f"🔍 Detected Emotions: {', '.join(unique_labels)}")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_emotion_sunburst(df, ec),
                            use_container_width=True, key=f"em_sun_{key_prefix}")
        with c2:
            st.plotly_chart(create_emotion_by_user_chart(df, ec),
                            use_container_width=True, key=f"em_user_{key_prefix}")
        ui_divider()
        st.plotly_chart(create_emotion_timeline(df, ec),
                        use_container_width=True, key=f"em_timeline_{key_prefix}")
    else:
        st.warning("No emotion data available.")
        st.info("Enable 'Include Emotion Detection' in sidebar.")


def render_model_performance_tab(filtered_df, analysis_method, selected_language,
                                  enable_lang_detection, enable_emoji_analysis,
                                  enable_multimodal_fusion, sentiment_col):
    is_dark = st.session_state.get("theme", "dark") == "dark"
    text_color = "#e6edf3" if is_dark else "#1a1a2e"
    card_bg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
    border = "rgba(88,166,255,0.20)" if is_dark else "rgba(99,102,241,0.20)"
    header_bg = "linear-gradient(135deg,#58a6ff,#bc8cff)" if is_dark else "linear-gradient(135deg,#6366f1,#8b5cf6)"
    row_hover = "rgba(88,166,255,0.06)" if is_dark else "rgba(99,102,241,0.05)"

    st.markdown(f"""
    <div style="background:{card_bg};border:1px solid {border};border-radius:20px;
                padding:28px;margin-bottom:24px;backdrop-filter:blur(20px);">
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:{text_color};margin-bottom:20px;">
            🤖 Model Comparison
        </div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:{header_bg};">
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;
                               font-weight:700;text-align:left;border-radius:8px 0 0 0;">Feature</th>
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;
                               font-weight:700;text-align:left;">⚡ VADER</th>
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;
                               font-weight:700;text-align:left;border-radius:0 8px 0 0;">🤖 Transformer</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid {border};">
                    <td style="padding:12px 16px;color:{text_color};font-weight:600;font-size:0.85rem;">Type</td>
                    <td style="padding:12px 16px;color:#3fb950;font-size:0.85rem;">Rule-Based</td>
                    <td style="padding:12px 16px;color:#bc8cff;font-size:0.85rem;">Deep Learning</td>
                </tr>
                <tr style="border-bottom:1px solid {border};background:{row_hover};">
                    <td style="padding:12px 16px;color:{text_color};font-weight:600;font-size:0.85rem;">Speed</td>
                    <td style="padding:12px 16px;color:#fbbf24;font-size:0.85rem;">⚡ Very Fast</td>
                    <td style="padding:12px 16px;color:#f78166;font-size:0.85rem;">🐢 Slower</td>
                </tr>
                <tr style="border-bottom:1px solid {border};">
                    <td style="padding:12px 16px;color:{text_color};font-weight:600;font-size:0.85rem;">Sentiments</td>
                    <td style="padding:12px 16px;color:#58a6ff;font-size:0.85rem;">3 classes</td>
                    <td style="padding:12px 16px;color:#58a6ff;font-size:0.85rem;">7 emotions</td>
                </tr>
                <tr style="border-bottom:1px solid {border};background:{row_hover};">
                    <td style="padding:12px 16px;color:{text_color};font-weight:600;font-size:0.85rem;">Language</td>
                    <td style="padding:12px 16px;color:#3fb950;font-size:0.85rem;">English + Emoji</td>
                    <td style="padding:12px 16px;color:#3fb950;font-size:0.85rem;">Multilingual</td>
                </tr>
                <tr>
                    <td style="padding:12px 16px;color:{text_color};font-weight:600;font-size:0.85rem;">Best For</td>
                    <td style="padding:12px 16px;color:#bc8cff;font-size:0.85rem;">Quick Analysis</td>
                    <td style="padding:12px 16px;color:#bc8cff;font-size:0.85rem;">Deep Insights</td>
                </tr>
            </tbody>
        </table>
    </div>
    <div style="background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.25);
                border-left:4px solid #58a6ff;border-radius:10px;padding:14px 18px;">
        <span style="color:#58a6ff;font-weight:600;font-size:0.9rem;">
            💡 Transformer detects 7 emotions vs VADER's 3 sentiment classes
        </span>
    </div>
    """, unsafe_allow_html=True)
    # Model Agreement Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    is_dark2 = st.session_state.get("theme", "dark") == "dark"
    tc = "#e6edf3" if is_dark2 else "#1a1a2e"
    cbg = "rgba(22,27,34,0.95)" if is_dark2 else "rgba(255,255,255,0.95)"
    br = "rgba(88,166,255,0.20)" if is_dark2 else "rgba(99,102,241,0.20)"
    if "Multimodal" in analysis_method and "transformer_sentiment" in filtered_df.columns and "mm_sentiment" in filtered_df.columns:
        vader_labels = filtered_df["transformer_sentiment"].str.capitalize()
        model_labels = filtered_df["mm_sentiment"].str.capitalize()
        agree = (vader_labels == model_labels)
        agree_pct = agree.mean() * 100
        disagree_pct = 100 - agree_pct
        both_pos = ((vader_labels == "Positive") & (model_labels == "Positive")).sum()
        both_neg = ((vader_labels == "Negative") & (model_labels == "Negative")).sum()
        both_neu = ((vader_labels == "Neutral") & (model_labels == "Neutral")).sum()
        total = len(filtered_df)
        st.markdown(f"""
        <div style="background:{cbg};border:1px solid {br};border-radius:20px;padding:28px;margin-bottom:24px;">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:{tc};margin-bottom:20px;">
                🔀 Transformer vs Multimodal Agreement
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
                <div style="background:rgba(63,185,80,0.10);border-top:3px solid #3fb950;border-radius:14px;padding:20px;text-align:center;">
                    <div style="font-size:2.5rem;font-weight:700;color:#3fb950;">{agree_pct:.1f}%</div>
                    <div style="color:{tc};font-size:0.85rem;margin-top:6px;">✅ Models Agree</div>
                </div>
                <div style="background:rgba(247,129,102,0.10);border-top:3px solid #f78166;border-radius:14px;padding:20px;text-align:center;">
                    <div style="font-size:2.5rem;font-weight:700;color:#f78166;">{disagree_pct:.1f}%</div>
                    <div style="color:{tc};font-size:0.85rem;margin-top:6px;">❌ Models Disagree</div>
                </div>
            </div>
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr style="background:linear-gradient(135deg,#58a6ff,#bc8cff);">
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;text-align:left;">Sentiment</th>
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;text-align:left;">Both Agree</th>
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;text-align:left;">% of Total</th>
                </tr></thead>
                <tbody>
                    <tr><td style="padding:12px 16px;color:#3fb950;font-weight:600;">✅ Positive</td><td style="padding:12px 16px;color:{tc};">{both_pos} messages</td><td style="padding:12px 16px;color:{tc};">{both_pos/total*100:.1f}%</td></tr>
                    <tr><td style="padding:12px 16px;color:#58a6ff;font-weight:600;">😐 Neutral</td><td style="padding:12px 16px;color:{tc};">{both_neu} messages</td><td style="padding:12px 16px;color:{tc};">{both_neu/total*100:.1f}%</td></tr>
                    <tr><td style="padding:12px 16px;color:#f78166;font-weight:600;">❌ Negative</td><td style="padding:12px 16px;color:{tc};">{both_neg} messages</td><td style="padding:12px 16px;color:{tc};">{both_neg/total*100:.1f}%</td></tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        disagree_df = filtered_df[vader_labels != model_labels][["datetime","author","message"]].copy()
        disagree_df["Transformer"] = vader_labels[vader_labels != model_labels]
        disagree_df["Multimodal"] = model_labels[vader_labels != model_labels]
        if not disagree_df.empty:
            rows_html = ""
            for i, row in disagree_df.head(5).iterrows():
                rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                v_color = "#3fb950" if row["Transformer"]=="Positive" else ("#f78166" if row["Transformer"]=="Negative" else "#58a6ff")
                m_color = "#3fb950" if row["Multimodal"]=="Positive" else ("#f78166" if row["Multimodal"]=="Negative" else "#58a6ff")
                msg = str(row["message"])[:60] + "..." if len(str(row["message"]))>60 else str(row["message"])
                rows_html += f'''<tr style="background:{rbg};"><td style="padding:9px 12px;color:{tc};font-size:0.8rem;border-bottom:1px solid {br};">{row["author"]}</td><td style="padding:9px 12px;color:{tc};font-size:0.78rem;border-bottom:1px solid {br};">{msg}</td><td style="padding:9px 12px;border-bottom:1px solid {br};"><span style="color:{v_color};font-weight:700;">{row["Transformer"]}</span></td><td style="padding:9px 12px;border-bottom:1px solid {br};"><span style="color:{m_color};font-weight:700;">{row["Multimodal"]}</span></td></tr>'''
            st.markdown(f"""
            <div style="background:{cbg};border:1px solid {br};border-radius:14px;overflow:hidden;margin-bottom:24px;">
                <div style="padding:16px 20px;font-weight:700;color:{tc};">⚡ Where Models Disagreed</div>
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr style="background:linear-gradient(135deg,#f78166,#bc8cff);">
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">👤 User</th>
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">💬 Message</th>
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">🤖 Transformer</th>
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">🌐 Multimodal</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
    elif "vader_compound" in filtered_df.columns and "sentiment_score" in filtered_df.columns:
        vader_labels = filtered_df["vader_compound"].apply(
            lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
        model_labels = filtered_df["sentiment"].str.strip().str.capitalize()
        agree = (vader_labels == model_labels)
        agree_pct = agree.mean() * 100
        disagree_pct = 100 - agree_pct
        both_pos = ((vader_labels == "Positive") & (model_labels == "Positive")).sum()
        both_neg = ((vader_labels == "Negative") & (model_labels == "Negative")).sum()
        both_neu = ((vader_labels == "Neutral") & (model_labels == "Neutral")).sum()
        total = len(filtered_df)
        st.markdown(f"""
        <div style="background:{cbg};border:1px solid {br};border-radius:20px;padding:28px;margin-bottom:24px;">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:{tc};margin-bottom:20px;">
                {"🔀 Transformer vs Multimodal Agreement" if "Multimodal" in analysis_method else "📊 VADER vs Transformer Agreement"}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
                <div style="background:rgba(63,185,80,0.10);border-top:3px solid #3fb950;border-radius:14px;padding:20px;text-align:center;">
                    <div style="font-size:2.5rem;font-weight:700;color:#3fb950;">{agree_pct:.1f}%</div>
                    <div style="color:{tc};font-size:0.85rem;margin-top:6px;">✅ Models Agree</div>
                </div>
                <div style="background:rgba(247,129,102,0.10);border-top:3px solid #f78166;border-radius:14px;padding:20px;text-align:center;">
                    <div style="font-size:2.5rem;font-weight:700;color:#f78166;">{disagree_pct:.1f}%</div>
                    <div style="color:{tc};font-size:0.85rem;margin-top:6px;">❌ Models Disagree</div>
                </div>
            </div>
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr style="background:linear-gradient(135deg,#58a6ff,#bc8cff);">
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;text-align:left;">Sentiment</th>
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;text-align:left;">Both Agree</th>
                    <th style="padding:12px 16px;color:white;font-size:0.82rem;text-align:left;">% of Total</th>
                </tr></thead>
                <tbody>
                    <tr><td style="padding:12px 16px;color:#3fb950;font-weight:600;">✅ Positive</td><td style="padding:12px 16px;color:{tc};">{both_pos} messages</td><td style="padding:12px 16px;color:{tc};">{both_pos/total*100:.1f}%</td></tr>
                    <tr><td style="padding:12px 16px;color:#58a6ff;font-weight:600;">😐 Neutral</td><td style="padding:12px 16px;color:{tc};">{both_neu} messages</td><td style="padding:12px 16px;color:{tc};">{both_neu/total*100:.1f}%</td></tr>
                    <tr><td style="padding:12px 16px;color:#f78166;font-weight:600;">❌ Negative</td><td style="padding:12px 16px;color:{tc};">{both_neg} messages</td><td style="padding:12px 16px;color:{tc};">{both_neg/total*100:.1f}%</td></tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        disagree_df = filtered_df[vader_labels != model_labels][["datetime","author","message"]].copy()
        disagree_df["VADER"] = vader_labels[vader_labels != model_labels]
        disagree_df["Model"] = model_labels[vader_labels != model_labels]
        if not disagree_df.empty:
            rows_html = ""
            for i, row in disagree_df.head(5).iterrows():
                rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                v_color = "#3fb950" if row["VADER"]=="Positive" else ("#f78166" if row["VADER"]=="Negative" else "#58a6ff")
                m_color = "#3fb950" if row["Model"]=="Positive" else ("#f78166" if row["Model"]=="Negative" else "#58a6ff")
                msg = str(row["message"])[:60] + "..." if len(str(row["message"]))>60 else str(row["message"])
                rows_html += f'''<tr style="background:{rbg};"><td style="padding:9px 12px;color:{tc};font-size:0.8rem;border-bottom:1px solid {br};">{row["author"]}</td><td style="padding:9px 12px;color:{tc};font-size:0.78rem;border-bottom:1px solid {br};">{msg}</td><td style="padding:9px 12px;border-bottom:1px solid {br};"><span style="color:{v_color};font-weight:700;">{row["VADER"]}</span></td><td style="padding:9px 12px;border-bottom:1px solid {br};"><span style="color:{m_color};font-weight:700;">{row["Model"]}</span></td></tr>'''
            st.markdown(f"""
            <div style="background:{cbg};border:1px solid {br};border-radius:14px;overflow:hidden;margin-bottom:24px;">
                <div style="padding:16px 20px;font-weight:700;color:{tc};">⚡ Where Models Disagreed</div>
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr style="background:linear-gradient(135deg,#f78166,#bc8cff);">
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">👤 User</th>
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">💬 Message</th>
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">⚡ VADER</th>
                        <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">🤖 Model</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run Transformer or Multimodal analysis to see model agreement.")
    # Confusion Matrix
    st.markdown("<br>", unsafe_allow_html=True)
    import plotly.figure_factory as ff
    import numpy as np
    is_dark3 = st.session_state.get("theme","dark") == "dark"
    tc3 = "#e6edf3" if is_dark3 else "#1a1a2e"
    cbg3 = "rgba(22,27,34,0.95)" if is_dark3 else "rgba(255,255,255,0.95)"
    br3 = "rgba(88,166,255,0.20)" if is_dark3 else "rgba(99,102,241,0.20)"

    if "Multimodal" in analysis_method:
        # Transformer vs Multimodal
        if "sentiment_score" in filtered_df.columns:
            if "transformer_sentiment" in filtered_df.columns:
                trans_labels = filtered_df["transformer_sentiment"].str.capitalize()
            elif "vader_compound" in filtered_df.columns:
                trans_labels = filtered_df["vader_compound"].apply(
                    lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
            else:
                trans_labels = filtered_df["sentiment_score"].apply(
                    lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
            if "mm_sentiment" in filtered_df.columns:
                mm_labels = filtered_df["mm_sentiment"].str.capitalize()
            else:
                import emoji as emoji_lib
                def get_mm_label(row):
                    msg = str(row.get("message",""))
                    emojis = [e["emoji"] for e in emoji_lib.emoji_list(msg)]
                    pos_e = {"😊","😂","😍","👍","🎉","❤️","🔥","💪","✅","🙌","😁","💯","😄"}
                    neg_e = {"😠","😢","👎","💔","😡","😭","🤦","😤","💀","🙄","😒","😞","😔"}
                    has_pos = any(e in pos_e for e in emojis)
                    has_neg = any(e in neg_e for e in emojis)
                    base = row.get("sentiment", "Neutral")
                    if has_neg and base == "Positive":
                        return "Neutral"
                    if has_pos and base == "Negative":
                        return "Neutral"
                    return base
                mm_labels = filtered_df.apply(get_mm_label, axis=1)
            labels = ["Positive", "Neutral", "Negative"]
            cm = []
            for true_label in labels:
                row = []
                for pred_label in labels:
                    count = ((trans_labels == true_label) & (mm_labels == pred_label)).sum()
                    row.append(int(count))
                cm.append(row)
            cm_array = np.array(cm)
            fig_cm = ff.create_annotated_heatmap(
                z=cm_array,
                x=[f"Multimodal: {l}" for l in labels],
                y=[f"Transformer: {l}" for l in labels],
                colorscale="Purples", showscale=True)
            fig_cm.update_layout(
                title={"text": "Confusion Matrix — Transformer vs Multimodal",
                       "font": {"size":17,"family":"Playfair Display","color":tc3}},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family":"DM Sans","color":tc3},
                margin=dict(l=40,r=40,t=55,b=40))
            st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix_mm")
            total2 = cm_array.sum()
            correct2 = cm_array.diagonal().sum()
            accuracy2 = correct2/total2*100 if total2>0 else 0
            st.markdown(f"""
            <div style="background:{cbg3};border:1px solid {br3};border-radius:16px;padding:20px;margin-top:16px;">
                <div style="font-weight:700;color:{tc3};margin-bottom:14px;">📊 Transformer vs Multimodal Agreement</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
                    <div style="background:rgba(63,185,80,0.10);border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:700;color:#3fb950;">{accuracy2:.1f}%</div>
                        <div style="color:{tc3};font-size:0.82rem;margin-top:4px;">Models Agree</div>
                    </div>
                    <div style="background:rgba(88,166,255,0.10);border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:700;color:#58a6ff;">{correct2}</div>
                        <div style="color:{tc3};font-size:0.82rem;margin-top:4px;">Messages Agreed</div>
                    </div>
                    <div style="background:rgba(247,129,102,0.10);border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:700;color:#f78166;">{total2-correct2}</div>
                        <div style="color:{tc3};font-size:0.82rem;margin-top:4px;">Messages Disagreed</div>
                    </div>
                </div>
                <div style="margin-top:12px;padding:10px;background:rgba(188,140,255,0.08);border-radius:8px;border-left:3px solid #bc8cff;">
                    <span style="color:#bc8cff;font-size:0.85rem;font-weight:600;">
                        💡 Multimodal adds emoji context to Transformer — disagreements show where emojis changed the sentiment
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run Multimodal analysis to see confusion matrix.")

    else:
        # VADER vs Transformer
        if "vader_compound" in filtered_df.columns:
            vader_labels2 = filtered_df["vader_compound"].apply(
                lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
            model_labels2 = filtered_df["sentiment"].str.capitalize()
            labels = ["Positive", "Neutral", "Negative"]
            cm = []
            for true_label in labels:
                row = []
                for pred_label in labels:
                    count = ((vader_labels2 == true_label) & (model_labels2 == pred_label)).sum()
                    row.append(int(count))
                cm.append(row)
            cm_array = np.array(cm)
            fig_cm = ff.create_annotated_heatmap(
                z=cm_array,
                x=[f"Transformer: {l}" for l in labels],
                y=[f"VADER: {l}" for l in labels],
                colorscale="Blues", showscale=True)
            fig_cm.update_layout(
                title={"text": "Confusion Matrix — VADER vs Transformer",
                       "font": {"size":17,"family":"Playfair Display","color":tc3}},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family":"DM Sans","color":tc3},
                margin=dict(l=40,r=40,t=55,b=40))
            st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix_trans")
            total2 = cm_array.sum()
            correct2 = cm_array.diagonal().sum()
            accuracy2 = correct2/total2*100 if total2>0 else 0
            cbg3 = "rgba(22,27,34,0.95)" if is_dark3 else "rgba(255,255,255,0.95)"
            br3 = "rgba(88,166,255,0.20)" if is_dark3 else "rgba(99,102,241,0.20)"
            st.markdown(f"""
            <div style="background:{cbg3};border:1px solid {br3};border-radius:16px;padding:20px;margin-top:16px;">
                <div style="font-weight:700;color:{tc3};margin-bottom:14px;">📊 Confusion Matrix Interpretation</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
                    <div style="background:rgba(63,185,80,0.10);border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:700;color:#3fb950;">{accuracy2:.1f}%</div>
                        <div style="color:{tc3};font-size:0.82rem;margin-top:4px;">Overall Agreement</div>
                    </div>
                    <div style="background:rgba(88,166,255,0.10);border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:700;color:#58a6ff;">{correct2}</div>
                        <div style="color:{tc3};font-size:0.82rem;margin-top:4px;">Messages Agreed</div>
                    </div>
                    <div style="background:rgba(247,129,102,0.10);border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:700;color:#f78166;">{total2-correct2}</div>
                        <div style="color:{tc3};font-size:0.82rem;margin-top:4px;">Messages Disagreed</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run Transformer or Multimodal analysis to see confusion matrix.")



def render_research_tab():
    is_dark = st.session_state.get("theme","dark") == "dark"
    tc = "#e6edf3" if is_dark else "#1a1a2e"
    cbg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
    br = "#30363d" if is_dark else "#e2e8f0"
    row1 = "rgba(30,40,60,0.6)" if is_dark else "#ffffff"
    row2 = "rgba(22,30,50,0.8)" if is_dark else "#f8f9ff"
    research_rows = [
        ("VADER", "Twitter Sentiment140", "68.0%", "0.67", "Hutto & Gilbert, 2014", "#3fb950"),
        ("VADER", "Amazon Reviews", "71.0%", "0.70", "Hutto & Gilbert, 2014", "#3fb950"),
        ("VADER", "Movie Reviews (IMDB)", "65.0%", "0.64", "Hutto & Gilbert, 2014", "#3fb950"),
        ("BERT (Transformer)", "SST-2 Stanford", "93.5%", "0.934", "Devlin et al., 2019", "#bc8cff"),
        ("RoBERTa (Transformer)", "SST-2 Stanford", "94.8%", "0.947", "Liu et al., 2019", "#bc8cff"),
        ("XLNet (Transformer)", "SST-2 Stanford", "95.1%", "0.950", "Yang et al., 2019", "#bc8cff"),
        ("Multimodal Text+Audio", "CMU-MOSI", "82.5%", "0.821", "Zadeh et al., 2018", "#58a6ff"),
        ("Multimodal Text+Image", "Hateful Memes", "84.7%", "0.843", "Kiela et al., 2020", "#58a6ff"),
        ("Multimodal Fusion", "CMU-MOSEI", "85.3%", "0.851", "Zadeh et al., 2018", "#58a6ff"),
    ]
    rows_html = ""
    for i, (model, dataset, acc, f1, ref, mc) in enumerate(research_rows):
        rbg = row2 if i%2==0 else row1
        rows_html += f"<tr style='background:{rbg};'><td style='padding:10px 14px;color:{mc};font-weight:700;font-size:0.82rem;border-bottom:1px solid {br};'>{model}</td><td style='padding:10px 14px;color:{tc};font-size:0.82rem;border-bottom:1px solid {br};'>{dataset}</td><td style='padding:10px 14px;color:#fbbf24;font-weight:700;font-size:0.85rem;border-bottom:1px solid {br};'>{acc}</td><td style='padding:10px 14px;color:{tc};font-size:0.82rem;border-bottom:1px solid {br};'>{f1}</td><td style='padding:10px 14px;color:#8b949e;font-size:0.78rem;font-style:italic;border-bottom:1px solid {br};'>{ref}</td></tr>"
    st.markdown(f'''
    <div style="background:{cbg};border:1px solid rgba(88,166,255,0.20);border-radius:20px;padding:28px;margin-top:8px;">
        <div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:{tc};margin-bottom:6px;">📚 Research Paper Benchmark Values</div>
        <div style="font-size:0.82rem;color:#8b949e;margin-bottom:20px;">Published accuracy values from peer-reviewed papers — cite these in your viva 🎓</div>
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;">
            <thead><tr style="background:linear-gradient(135deg,#58a6ff,#bc8cff);">
                <th style="padding:12px 14px;color:white;font-size:0.78rem;text-align:left;">🤖 Model</th>
                <th style="padding:12px 14px;color:white;font-size:0.78rem;text-align:left;">📊 Dataset</th>
                <th style="padding:12px 14px;color:white;font-size:0.78rem;text-align:left;">🎯 Accuracy</th>
                <th style="padding:12px 14px;color:white;font-size:0.78rem;text-align:left;">📈 F1 Score</th>
                <th style="padding:12px 14px;color:white;font-size:0.78rem;text-align:left;">📖 Reference</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
            <div style="background:rgba(63,185,80,0.10);border-top:3px solid #3fb950;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:700;color:#3fb950;">65-71%</div>
                <div style="color:{tc};font-size:0.82rem;margin-top:4px;">⚡ VADER Accuracy</div>
                <div style="color:#8b949e;font-size:0.75rem;margin-top:2px;">Rule-based, Fast</div>
            </div>
            <div style="background:rgba(188,140,255,0.10);border-top:3px solid #bc8cff;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:700;color:#bc8cff;">93-95%</div>
                <div style="color:{tc};font-size:0.82rem;margin-top:4px;">🤖 Transformer Accuracy</div>
                <div style="color:#8b949e;font-size:0.75rem;margin-top:2px;">Deep Learning</div>
            </div>
            <div style="background:rgba(88,166,255,0.10);border-top:3px solid #58a6ff;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:700;color:#58a6ff;">82-85%</div>
                <div style="color:{tc};font-size:0.82rem;margin-top:4px;">🌐 Multimodal Accuracy</div>
                <div style="color:#8b949e;font-size:0.75rem;margin-top:2px;">Text + Emoji Fusion</div>
            </div>
        </div>
        <div style="margin-top:16px;padding:12px 16px;background:rgba(88,166,255,0.08);border-radius:10px;border-left:3px solid #58a6ff;">
            <span style="color:#58a6ff;font-size:0.82rem;font-weight:600;">
                💡 Viva Tip: VADER (rule-based, 65-71%) → Transformer (deep learning, 93-95%) → Multimodal (fusion, 82-85%). Higher accuracy justifies computational cost of Transformers.
            </span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def render_vader_dashboard(df, sentiment_col):
    if df.empty:
        st.warning("No data available.")
        return
    df = df.copy()

    # Multilingual: use XLM-RoBERTa for non-English, VADER for English
    if "detected_language" not in df.columns:
        with st.spinner("🌍 Detecting languages..."):
            df["detected_language"] = df["message"].astype(str).apply(detect_text_language)

    xlm_model = load_multilingual_model()
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    compounds, sentiments = [], []
    for _, row in df.iterrows():
        lang = row.get("detected_language", "english")
        msg = str(row["message"])
        if lang in ["english"]:
            # English: use VADER
            c = vader.polarity_scores(msg)["compound"]
            compounds.append(c)
            sentiments.append("Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"))
        else:
            # Non-English / Hinglish / Urdu / Spanish: use XLM-RoBERTa
            sent, score = multilingual_sentiment(msg, xlm_model)
            compounds.append(score if sent == "Positive" else (-score if sent == "Negative" else 0.0))
            sentiments.append(sent)

    df["vader_compound"] = compounds
    df["sentiment"] = sentiments

    (chat_tab, overview_tab, url_tab, sentiment_tab, users_tab,
     words_tab, activity_tab, top_tab, all_tab, research_tab) = st.tabs([
        "📝 Summary", "📊 Overview", "🔗 URLs", "😊 Sentiment",
        "👥 Users", "☁️ Words", "📅 Activity", "🏆 Top", "💬 Messages", "📚 Research"
    ])

    with chat_tab:
        uc = df["author"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        with c1: ui_metric_card("Total Messages", f"{len(df):,}", "💬")
        with c2: ui_metric_card("Active Users", f"{df['author'].nunique():,}", "👥")
        with c3: ui_metric_card("Majority Sentiment", df[sentiment_col].mode()[0], "🗳️")
        with c4: ui_metric_card("Avg Sentiment", "Positive" if df["vader_compound"].mean() >= 0.05 else ("Negative" if df["vader_compound"].mean() <= -0.05 else "Neutral"), "📊")
        ui_divider()
        st.markdown(f"**📅 Date Range:** {df['datetime'].min().strftime('%Y-%m-%d')} → {df['datetime'].max().strftime('%Y-%m-%d')}")
        st.markdown(f"**🏆 Most Active:** `{uc.idxmax()}` · **{uc.max():,}** messages")
        if "emotion" in df.columns:
            st.markdown(f"**🎭 Overall Mood:** {df['emotion'].mode()[0].capitalize()}")
        ui_divider()
        # Most Emotional Day
        df_copy = df.copy()
        df_copy["date"] = df_copy["datetime"].dt.date
        daily_neg = df_copy[df_copy["sentiment"].str.lower()=="negative"].groupby("date").size()
        daily_pos = df_copy[df_copy["sentiment"].str.lower()=="positive"].groupby("date").size()
        is_dark_ved = st.session_state.get("theme","dark") == "dark"
        tc_ved = "#e6edf3" if is_dark_ved else "#1a1a2e"
        cbg_ved = "rgba(22,27,34,0.95)" if is_dark_ved else "rgba(255,255,255,0.95)"
        c1_ved, c2_ved = st.columns(2)
        with c1_ved:
            if not daily_neg.empty:
                most_neg_day = daily_neg.idxmax()
                most_neg_count = daily_neg.max()
                st.markdown(f"""
                <div style="background:{cbg_ved};border:1px solid #f7816630;border-top:3px solid #f78166;
                            border-radius:14px;padding:20px;margin-bottom:16px;">
                    <div style="font-size:0.72rem;color:#f78166;font-weight:700;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:8px;">😠 Most Negative Day</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;
                                color:#f78166;">{most_neg_day}</div>
                    <div style="font-size:0.85rem;color:#8b949e;margin-top:4px;">{most_neg_count} negative messages</div>
                </div>
                """, unsafe_allow_html=True)
        with c2_ved:
            if not daily_pos.empty:
                most_pos_day = daily_pos.idxmax()
                most_pos_count = daily_pos.max()
                st.markdown(f"""
                <div style="background:{cbg_ved};border:1px solid #3fb95030;border-top:3px solid #3fb950;
                            border-radius:14px;padding:20px;margin-bottom:16px;">
                    <div style="font-size:0.72rem;color:#3fb950;font-weight:700;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:8px;">😊 Most Positive Day</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;
                                color:#3fb950;">{most_pos_day}</div>
                    <div style="font-size:0.85rem;color:#8b949e;margin-top:4px;">{most_pos_count} positive messages</div>
                </div>
                """, unsafe_allow_html=True)
        ui_divider()
        st.success(generate_text_summary(df))

    with overview_tab:
        c1, c2, c3, c4 = st.columns(4)
        with c1: ui_metric_card("Total Messages", f"{len(df):,}", "💬")
        with c2: ui_metric_card("Active Users", f"{df['author'].nunique():,}", "👥")
        with c3: ui_metric_card("Positive", f"{len(df[df['sentiment']=='Positive']):,}", "✅")
        with c4: ui_metric_card("Negative", f"{len(df[df['sentiment']=='Negative']):,}", "❌")
        ui_divider()
        c1, c2 = st.columns(2)
        with c1:
            if "message" in df.columns:
                txt = df["message"].astype(str)
                txt = txt.str.replace(r"<Media omitted>|https?://\S+", "", regex=True)
                txt = txt.apply(lambda x: emoji.replace_emoji(x, replace=""))
                txt = txt.str.replace(r"[^\w\s]|\d+", "", regex=True)
                sw = set(stopwords.words("english"))
                wc_text = " ".join(txt.apply(lambda x: " ".join(
                    [w for w in x.split() if len(w) >= 2 and w not in sw])).tolist())
                if wc_text.strip():
                    try:
                        save_wordcloud(wc_text, "wc.png")
                        st.image("wc.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Word cloud error: {e}")
        with c2:
            st.plotly_chart(create_message_heatmap(df),
                            use_container_width=True, key="vader_overview_heatmap")
        ui_divider()
        # Compute user stats directly from df (which has vader sentiment)
        _is_dark = st.session_state.get("theme","dark") == "dark"
        _tc = "#e6edf3" if _is_dark else "#1a1a2e"
        _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
        _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
        _user_stats = []
        for author, grp in df.groupby("author"):
            total = len(grp)
            pos = len(grp[grp["sentiment"].str.lower()=="positive"])
            neu = len(grp[grp["sentiment"].str.lower()=="neutral"])
            neg = len(grp[grp["sentiment"].str.lower()=="negative"])
            avg = grp["vader_compound"].mean() if "vader_compound" in grp.columns else 0
            _user_stats.append({"Author": author, "Total": total, "Positive": pos, "Neutral": neu, "Negative": neg,
                                  "Pos%": f"{pos/total*100:.1f}%", "Neg%": f"{neg/total*100:.1f}%", "Avg Score": f"{avg:.3f}"})
        _hd_u = "".join([f"<th style='padding:10px 12px;color:white;font-size:0.75rem;font-weight:700;text-align:left;'>{c}</th>" for c in ["Author","Total","Positive","Neutral","Negative","Pos%","Neg%","Avg Score"]])
        _rows_u = ""
        for i, r in enumerate(_user_stats):
            _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
            _rows_u += f"<tr style='background:{_rbg};'><td style='padding:9px 12px;color:{_tc};font-weight:600;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Author']}</td><td style='padding:9px 12px;color:{_tc};font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Total']}</td><td style='padding:9px 12px;color:#3fb950;font-weight:600;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Positive']}</td><td style='padding:9px 12px;color:#58a6ff;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Neutral']}</td><td style='padding:9px 12px;color:#f78166;font-weight:600;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Negative']}</td><td style='padding:9px 12px;color:#3fb950;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Pos%']}</td><td style='padding:9px 12px;color:#f78166;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Neg%']}</td><td style='padding:9px 12px;color:#fbbf24;font-family:monospace;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Avg Score']}</td></tr>"
        st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:14px;overflow:hidden;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,#58a6ff,#bc8cff);'>{_hd_u}</tr></thead><tbody>{_rows_u}</tbody></table></div>", unsafe_allow_html=True)

    with url_tab:
        url_data = []
        for _, row in df.iterrows():
            msg = str(row.get("message_raw", row.get("message", "")))
            for url in re.findall(r"(https?://\S+|www\.\S+)", msg):
                url_data.append({"User": row["author"], "URL": url, "Date": row["datetime"].date()})
        if url_data:
            _is_dark = st.session_state.get("theme","dark") == "dark"
            _tc = "#e6edf3" if _is_dark else "#1a1a2e"
            _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
            _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
            udf_v = pd.DataFrame(url_data)
            ui_metric_card("Total URLs Found", len(udf_v), "🔗")
            vc_v = udf_v["User"].value_counts().reset_index(); vc_v.columns=["User","Count"]
            fig_url_v = px.bar(vc_v, x="User", y="Count", color="Count", color_continuous_scale="Blues")
            st.plotly_chart(apply_chart_theme(fig_url_v, "🔗 URLs per User"), use_container_width=True, key="vader_url_bar")
            rows_html = ""
            for i, r in udf_v.iterrows():
                u = r["URL"]; href = u if u.startswith("http") else "https://"+u
                _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                rows_html += f"<tr style='background:{_rbg};'><td style='padding:9px 14px;color:{_tc};font-size:0.82rem;border-bottom:1px solid {_br};'>{r['User']}</td><td style='padding:9px 14px;border-bottom:1px solid {_br};'><a href='{href}' target='_blank' style='color:#58a6ff;font-size:0.82rem;'>{u}</a></td><td style='padding:9px 14px;color:#8b949e;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Date']}</td></tr>"
            st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:14px;overflow:hidden;margin-top:16px;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,#58a6ff,#bc8cff);'><th style='padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;'>👤 User</th><th style='padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;'>🔗 URL</th><th style='padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;'>📅 Date</th></tr></thead><tbody>{rows_html}</tbody></table></div>", unsafe_allow_html=True)
        else:
            st.info("No URLs found.")

    with sentiment_tab:
        # Sentiment Timeline
        st.plotly_chart(create_sentiment_timeline(df, sentiment_col),
                        use_container_width=True, key="vader_sent_timeline")
        ui_divider()
        # Pie Chart + Avg Score
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_sentiment_pie_chart(df, sentiment_col),
                            use_container_width=True, key="vader_sent_pie")
        with c2:
            avg_score = df["vader_compound"].mean()
            pos_pct = len(df[df["sentiment"]=="Positive"])/len(df)*100
            neg_pct = len(df[df["sentiment"]=="Negative"])/len(df)*100
            neu_pct = len(df[df["sentiment"]=="Neutral"])/len(df)*100
            ui_metric_card("Avg VADER Score", f"{avg_score:.3f}", "📊")
            ui_metric_card("Positive %", f"{pos_pct:.1f}%", "✅")
            ui_metric_card("Negative %", f"{neg_pct:.1f}%", "❌")
            ui_metric_card("Neutral %", f"{neu_pct:.1f}%", "😐")
        ui_divider()
        # Sentiment by User
        st.plotly_chart(create_sentiment_comparison_chart(df, sentiment_col),
                        use_container_width=True, key="vader_sent_user")
        ui_divider()
        # Daily Compound Trend
        st.plotly_chart(create_daily_average_compound_chart(df),
                        use_container_width=True, key="vader_daily_avg")

    with users_tab:
        ul = df["author"].dropna().astype(str).unique().tolist()
        ul.sort()
        su = st.selectbox("Select user", ["Select User"] + ul, key="vader_user_select")
        if su != "Select User":
            udf = df[df["author"] == su]
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(create_user_activity_gauge(df, su),
                                use_container_width=True, key=f"gauge_{su}")
            with c2:
                ui_metric_card("Messages", len(udf), "💬")
                ui_metric_card("Avg Length", f"{udf['message'].str.len().mean():.0f} chars", "📏")
        ui_divider()
        _is_dark = st.session_state.get("theme","dark") == "dark"
        _tc = "#e6edf3" if _is_dark else "#1a1a2e"
        _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
        _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
        _user_stats2 = []
        for author, grp in df.groupby("author"):
            total = len(grp)
            pos = len(grp[grp["sentiment"].str.lower()=="positive"])
            neu = len(grp[grp["sentiment"].str.lower()=="neutral"])
            neg = len(grp[grp["sentiment"].str.lower()=="negative"])
            avg = grp["vader_compound"].mean() if "vader_compound" in grp.columns else 0
            _user_stats2.append({"Author": author, "Total": total, "Positive": pos, "Neutral": neu, "Negative": neg,
                                   "Pos%": f"{pos/total*100:.1f}%", "Neg%": f"{neg/total*100:.1f}%", "Avg Score": f"{avg:.3f}"})
        _hd_u2 = "".join([f"<th style='padding:10px 12px;color:white;font-size:0.75rem;font-weight:700;text-align:left;'>{c}</th>" for c in ["Author","Total","Positive","Neutral","Negative","Pos%","Neg%","Avg Score"]])
        _rows_u2 = ""
        for i, r in enumerate(_user_stats2):
            _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
            _rows_u2 += f"<tr style='background:{_rbg};'><td style='padding:9px 12px;color:{_tc};font-weight:600;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Author']}</td><td style='padding:9px 12px;color:{_tc};font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Total']}</td><td style='padding:9px 12px;color:#3fb950;font-weight:600;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Positive']}</td><td style='padding:9px 12px;color:#58a6ff;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Neutral']}</td><td style='padding:9px 12px;color:#f78166;font-weight:600;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['Negative']}</td><td style='padding:9px 12px;color:#3fb950;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Pos%']}</td><td style='padding:9px 12px;color:#f78166;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Neg%']}</td><td style='padding:9px 12px;color:#fbbf24;font-family:monospace;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['Avg Score']}</td></tr>"
        st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:14px;overflow:hidden;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,#58a6ff,#bc8cff);'>{_hd_u2}</tr></thead><tbody>{_rows_u2}</tbody></table></div>", unsafe_allow_html=True)

    with words_tab:
        c1, c2 = st.columns(2)
        with c1:
            if "message_raw" in df.columns:
                wc_text = " ".join(df.apply(clean_text_for_wordcloud, axis=1).tolist())
                if wc_text.strip():
                    try:
                        save_wordcloud(wc_text, "wc.png")
                        st.image("wc.png", use_container_width=True)
                    except: pass
        with c2:
            st.plotly_chart(create_word_frequency_chart(df, 15),
                            use_container_width=True, key="vader_words_freq")

    with activity_tab:
        c1, c2 = st.columns(2)
        with c1:
            dow = df["datetime"].dt.day_name().value_counts()
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow = dow.reindex([d for d in day_order if d in dow.index], fill_value=0)
            fig = go.Figure([go.Bar(x=dow.index, y=dow.values,
                                    marker=dict(color=dow.values, colorscale="Viridis"))])
            st.plotly_chart(apply_chart_theme(fig, "📅 Messages by Day"),
                            use_container_width=True, key="vader_dow")
        with c2:
            hr = df["datetime"].dt.hour.value_counts().sort_index()
            fig = go.Figure([go.Bar(x=[f"{h:02d}:00" for h in hr.index], y=hr.values,
                                    marker=dict(color=hr.values, colorscale="Plasma"))])
            st.plotly_chart(apply_chart_theme(fig, "⏰ Messages by Hour"),
                            use_container_width=True, key="vader_hour")
        ui_divider()
        st.plotly_chart(create_hourly_average_compound_chart(df),
                        use_container_width=True, key="vader_hourly_avg")

    with top_tab:
        if "vader_compound" in df.columns:
            _is_dark = st.session_state.get("theme","dark") == "dark"
            _tc = "#e6edf3" if _is_dark else "#1a1a2e"
            _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
            _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
            c1, c2 = st.columns(2)
            for col, title, asc_val, key_sfx, hdr_color in [
                (c1, "👍 Top 5 Most Positive", False, "pos", "#3fb950"),
                (c2, "👎 Top 5 Most Negative", True, "neg", "#f78166")
            ]:
                with col:
                    st.markdown(f"<h5 style='color:{_tc};margin-bottom:12px;'>{title}</h5>", unsafe_allow_html=True)
                    _tdf = df.sort_values("vader_compound", ascending=asc_val).head(5)[["datetime","author","message","vader_compound"]]
                    _rows_t = ""
                    for i, r in _tdf.iterrows():
                        _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                        score_color = "#3fb950" if r["vader_compound"]>0 else ("#f78166" if r["vader_compound"]<0 else "#8b949e")
                        _rows_t += f"<tr style='background:{_rbg};'><td style='padding:8px;color:#8b949e;font-size:0.75rem;border-bottom:1px solid {_br};'>{str(r['datetime'])[:16]}</td><td style='padding:8px;color:{_tc};font-size:0.8rem;font-weight:600;border-bottom:1px solid {_br};'>{r['author']}</td><td style='padding:8px;color:{_tc};font-size:0.78rem;border-bottom:1px solid {_br};'>{str(r['message'])[:60]}...</td><td style='padding:8px;color:{score_color};font-weight:700;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['vader_compound']:.3f}</td></tr>"
                    st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:12px;overflow:hidden;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,{hdr_color},{hdr_color}99);'><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Date</th><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Author</th><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Message</th><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Score</th></tr></thead><tbody>{_rows_t}</tbody></table></div>", unsafe_allow_html=True)

    with research_tab:
        render_research_tab()

    with all_tab:
        _is_dark = st.session_state.get("theme","dark") == "dark"
        _tc = "#e6edf3" if _is_dark else "#1a1a2e"
        _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
        _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
        c1, c2, c3 = st.columns(3)
        with c1: lim = st.number_input("Messages", 10, 1000, 50, 10, key="vader_all_lim")
        with c2: sb = st.selectbox("Sort by", ["datetime", sentiment_col, "vader_compound"], key="vader_sort_by")
        with c3: asc = st.checkbox("Ascending", False, key="vader_sort_asc")
        dcols = [c for c in ["datetime","author","message",sentiment_col,"vader_compound"] if c in df.columns]
        _adf = df[dcols].sort_values(sb, ascending=asc).head(lim)
        SENT_COLORS_V = {"positive":("#166534","#ffffff"),"negative":("#7f1d1d","#ffffff"),"neutral":("#1e3a5f","#ffffff")}
        _hd_a = "".join([f"<th style='padding:10px 12px;color:white;font-size:0.75rem;font-weight:700;text-align:left;'>{c}</th>" for c in _adf.columns])
        _rows_a = ""
        for i, r in _adf.iterrows():
            _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
            cells = ""
            for c, v in zip(_adf.columns, r.values):
                if c == sentiment_col:
                    s = str(v).lower()
                    sbg, sfg = SENT_COLORS_V.get(s, ("#1e3a5f","#ffffff"))
                    cells += f"<td style='padding:8px 12px;border-bottom:1px solid {_br};'><span style='background:{sbg};color:{sfg};padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;'>{v}</span></td>"
                elif c == "vader_compound":
                    sc = "#3fb950" if float(v)>0 else ("#f78166" if float(v)<0 else "#8b949e")
                    cells += f"<td style='padding:8px 12px;colossssssr:{sc};font-weight:700;font-size:0.82rem;border-bottom:1px solid {_br};'>{v}</td>"
                else:
                    cells += f"<td style='padding:8px 12px;color:{_tc};font-size:0.8rem;border-bottom:1px solid {_br};'>{str(v)[:60] if len(str(v))>60 else v}</td>"
            _rows_a += f"<tr style='background:{_rbg};'>{cells}</tr>"
        st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:14px;overflow:hidden;overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,#58a6ff,#bc8cff);'>{_hd_a}</tr></thead><tbody>{_rows_a}</tbody></table></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#                  SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2.8rem; filter:drop-shadow(0 0 14px rgba(88,166,255,0.5));
                    animation: none;">💬</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.05rem;
                    font-weight:700; color:#c9d1d9; margin-top:6px;">
            WhatsApp Analyzer
        </div>
        <div style="font-size:0.65rem; color:#8b949e; margin-top:3px; letter-spacing:1px;">
            B.TECH FINAL YEAR PROJECT
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    theme_icon = "☀️ Switch to Light" if st.session_state.theme == "dark" else "🌙 Switch to Dark"
    if st.button(theme_icon, key="theme_toggle_btn", use_container_width=True):
        # Save current analysis method before rerun
        if "main_analysis_method" in st.session_state:
            options = [
                "VADER - Text Sentiment Analysis (Baseline Model)",
                "Transformers (Advanced)",
                "Multimodal (Text + Emoji)"
            ]
            current = st.session_state.get("main_analysis_method", options[0])
            st.session_state.analysis_method_index = options.index(current) if current in options else 0
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # Data Source
    st.markdown('<div class="sb-label">📁 Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload WhatsApp Chat Export",
        type=["txt"], key="chat_file_uploader"
    )
    if uploaded_file is not None:
        st.session_state["raw_chat_bytes"] = uploaded_file.getvalue()
        st.session_state["chat_filename"] = uploaded_file.name

    if "raw_chat_bytes" not in st.session_state:
        st.info("👆 Upload a WhatsApp .txt export to begin")
        st.markdown("""
**📋 How to Export:**
1. Open WhatsApp chat
2. Tap ⋮ → Export chat
3. Choose Without media
4. Upload the .txt file
        """)

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # Analysis Engine
    st.markdown('<div class="sb-label">⚙️ Analysis Engine</div>', unsafe_allow_html=True)
    analysis_method = st.radio(
        "Select Methodology",
        options=[
            "VADER - Text Sentiment Analysis (Baseline Model)",
            "Transformers (Advanced)",
            "Multimodal (Text + Emoji)"
        ],
        index=st.session_state.get("analysis_method_index", 0),
        key="main_analysis_method"
    )
    options_list = [
        "VADER - Text Sentiment Analysis (Baseline Model)",
        "Transformers (Advanced)",
        "Multimodal (Text + Emoji)"
    ]
    if analysis_method in options_list:
        st.session_state.analysis_method_index = options_list.index(analysis_method)

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    manual_lang = "Auto"
    lang_mode = "🤖 Auto-Detect (Recommended)"
    st.session_state["lang_mode"] = lang_mode
    st.session_state["manual_lang"] = manual_lang

    if analysis_method in ["Transformers (Advanced)", "Multimodal (Text + Emoji)"]:
        st.checkbox("Include Emotion Detection", key="include_emotion",
                    value=st.session_state.get("include_emotion", True))
    else:
        st.session_state.include_emotion = False

    selected_language = manual_lang.lower() if manual_lang != "Auto" else "english"
    enable_lang_detection = (lang_mode == "🤖 Auto-Detect (Recommended)")
    enable_emoji_analysis = True
    enable_multimodal_fusion = True
    if analysis_method == "Multimodal (Text + Emoji)" and MULTIMODAL_AVAILABLE:
        with st.expander("⚙️ Multimodal Settings", expanded=False):
            enable_emoji_analysis = st.checkbox("Emoji Sentiment Analysis", True, key="mm_emoji_analysis")
            enable_multimodal_fusion = st.checkbox("Multimodal Fusion", True, key="mm_fusion")


# ════════════════════════════════════════════
#              DATA PROCESSING
# ════════════════════════════════════════════
@st.cache_data
def get_preprocessed_df(bytes_data):
    try:
        try:
            string_data = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            string_data = bytes_data.decode("latin-1")
        df = parse_chat(string_data)
        if df.empty:
            return pd.DataFrame()
        df = preprocess_df(df)
        df["user"] = df["author"]
        if "is_system_message" in df.columns:
            df = df[~df["is_system_message"]].reset_index(drop=True)
        df = df[df["message"].str.strip() != ""].reset_index(drop=True)
        df["length"] = df["message"].str.len()
        return df
    except Exception as e:
        logger.error(f"Error: {e}")
        return pd.DataFrame()


@st.cache_data
def process_data(bytes_data, _analysis_method, _include_emotions=False,
                 _language="english", _enable_lang_detection=False,
                 _enable_emoji_analysis=False, _enable_multimodal_fusion=False,
                 _transformer_language="English"):
    try:
        df = get_preprocessed_df(bytes_data)
        if df.empty:
            st.error("Chat file is empty or invalid format.")
            return pd.DataFrame()
        df = df.copy()
        df["sentiment"] = "Neutral"
        df["sentiment_score"] = 0.0
        df["vader_compound"] = 0.0

        if "Transformers" in _analysis_method:
            # Always use XLM-RoBERTa multilingual model
            with st.spinner("🌍 Running multilingual sentiment analysis..."):
                df = apply_multilingual_analysis(df, show_progress=True, include_lang_col=True)
                df["vader_compound"] = df["sentiment_score"]
            # Emotion detection
            if _include_emotions:
                analyzer = load_advanced_sentiment_analyzer(
                    sentiment_model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    emotion_model="j-hartmann/emotion-english-distilroberta-base")
                if analyzer:
                    with st.spinner("🎭 Detecting emotions..."):
                        emotions = []
                        for t in df["message"].fillna("").astype(str).tolist():
                            em, _ = predict_emotion(t, analyzer)
                            emotions.append(em)
                        df["transformer_emotion"] = emotions
                        df["emotion"] = emotions

        elif "Multimodal" in _analysis_method and MULTIMODAL_AVAILABLE:
            with st.spinner("🌍 Running multilingual multimodal analysis..."):
                # Step 1: Run Transformer first and save results
                df = apply_multilingual_analysis(df, show_progress=False, include_lang_col=True)
                df["transformer_sentiment"] = df["sentiment"].str.capitalize()
                df["transformer_score"] = df["sentiment_score"]
                df["vader_compound"] = df["sentiment_score"]
                try:
                    # Save transformer results before multimodal overwrites
                    _trans_sent = df["transformer_sentiment"].copy()
                    _trans_score = df["transformer_score"].copy()
                    # Step 2: Run Multimodal on top
                    df = apply_multimodal_sentiment(
                        df, language=_language,
                        enable_language_detection=_enable_lang_detection,
                        enable_emoji_analysis=_enable_emoji_analysis,
                        enable_multimodal=_enable_multimodal_fusion,
                        include_emotion=_include_emotions)
                    # Restore transformer results after multimodal
                    df["transformer_sentiment"] = _trans_sent.values
                    df["transformer_score"] = _trans_score.values
                    if "mm_sentiment" in df.columns:
                        df["sentiment"] = df["mm_sentiment"]
                    else:
                        df["mm_sentiment"] = df["sentiment"]
                    if "mm_score" in df.columns:
                        df["sentiment_score"] = df["mm_score"]
                    if "mm_emotion" in df.columns:
                        df["emotion"] = df["mm_emotion"]
                except Exception:
                    # Fallback to multilingual XLM
                    df["sentiment"] = df["transformer_sentiment"]
                    df["sentiment_score"] = df["transformer_score"]

        elif "VADER" in _analysis_method:
            with st.spinner("Running VADER analysis..."):
                try:
                    df_vader_results = apply_vader(df.copy())
                    if isinstance(df_vader_results, pd.DataFrame):
                        if "sentiment" in df_vader_results.columns:
                            df["sentiment"] = df_vader_results["sentiment"].values
                        if "vader_compound" in df_vader_results.columns:
                            df["vader_compound"] = df_vader_results["vader_compound"].values
                        elif "compound" in df_vader_results.columns:
                            df["vader_compound"] = df_vader_results["compound"].values
                        df["sentiment_score"] = df["vader_compound"]
                    else:
                        raise ValueError("apply_vader returned non-DataFrame")
                except Exception:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    _v = SentimentIntensityAnalyzer()
                    df["vader_compound"] = df["message"].astype(str).apply(lambda x: _v.polarity_scores(x)["compound"])
                    df["sentiment"] = df["vader_compound"].apply(lambda s: "Positive" if s>=0.05 else ("Negative" if s<=-0.05 else "Neutral"))
                    df["sentiment_score"] = df["vader_compound"]

        if "emotion" not in df.columns:
            df["emotion"] = [detect_emotion(m) for m in df["message"]]
        df["length"] = df["message"].str.len()
        return df
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()


# ════════════════════════════════════════════
#               MAIN APP LOGIC
# ════════════════════════════════════════════
if "raw_chat_bytes" in st.session_state:
    method_short = ("VADER" if "VADER" in analysis_method
                    else ("Transformers" if "Transformers" in analysis_method
                    else "Multimodal"))
    ui_header_banner(subtitle=f"{method_short} Model Active")

    transformer_language = st.session_state.get("transformer_language_selector", "English")
    df = process_data(
        st.session_state["raw_chat_bytes"], analysis_method,
        st.session_state.get("include_emotion", False), selected_language,
        enable_lang_detection, enable_emoji_analysis, enable_multimodal_fusion,
        _transformer_language=transformer_language)

    if df.empty:
        st.warning("No valid messages found.")
        st.stop()

    sentiment_col = "sentiment"

    with st.sidebar:
        st.markdown('<div class="sb-label">📊 Filters</div>', unsafe_allow_html=True)
        min_date = df["datetime"].dt.date.min()
        max_date = df["datetime"].dt.date.max()
        date_range = st.date_input("Date Range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date, key="date_range_filter")
        user_list = sorted([u for u in df["author"].unique().tolist() if u is not None])
        selected_users = st.multiselect("Users", options=user_list,
                                         default=user_list, key="user_multiselect")
        sentiment_options = df[sentiment_col].unique().tolist()
        selected_sentiments = st.multiselect("Sentiments", options=sentiment_options,
                                              default=sentiment_options, key="sentiment_multiselect")
        length_range = st.slider("Message Length", 0, 500, (0, 500), key="length_slider")
        keyword = st.text_input("🔍 Search", placeholder="Keywords...", key="keyword_search")
        negative_only_mode = st.checkbox("⚠️ Negative Only", False, key="negative_only_mode")

    # Apply filters
    filtered_df = df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date = end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["datetime"].dt.date >= start_date) &
        (filtered_df["datetime"].dt.date <= end_date)]
    if selected_users:
        filtered_df = filtered_df[filtered_df["author"].isin(selected_users)]
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df[sentiment_col].isin(selected_sentiments)]
    min_len, max_len = length_range
    filtered_df = filtered_df[
        (filtered_df["length"] >= min_len) & (filtered_df["length"] <= max_len)]
    if keyword:
        filtered_df = filtered_df[
            filtered_df["message"].str.contains(keyword, case=False, na=False)]
    if negative_only_mode:
        filtered_df = filtered_df[filtered_df[sentiment_col].str.lower() == "negative"]

    with st.sidebar:
        st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
        st.download_button("📥 Download CSV",
                           filtered_df.to_csv(index=False).encode("utf-8"),
                           "filtered_chat.csv", "text/csv", key="sidebar_dl_csv")

    if filtered_df.empty:
        st.warning("No data matches current filters.")
        st.stop()

    # ── Pre-compute VADER sentiment for KPI accuracy ──
    if "VADER" in analysis_method:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
            filtered_df = filtered_df.copy()
            filtered_df["vader_compound"] = filtered_df["message"].astype(str).apply(
                lambda x: _vader.polarity_scores(x)["compound"])
            filtered_df["sentiment"] = filtered_df["vader_compound"].apply(
                lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral"))
            filtered_df["sentiment_score"] = filtered_df["vader_compound"]
        except Exception as e:
            st.warning(f"VADER pre-compute issue: {e}")



    # ── VADER Dashboard ──
    if "VADER" in analysis_method:
        render_vader_dashboard(filtered_df, sentiment_col)

    # ── Transformers Dashboard ──
    elif "Transformers" in analysis_method:
        st.markdown("<h1 style='text-align:center;'>🤖 Transformer Dashboard</h1>", unsafe_allow_html=True)
        tabs = st.tabs(["📝 Summary", "📊 Overview", "🔗 URLs", "😊 Sentiment",
                         "😊 Emotions", "📝 Emotion Words", "🎯 Confidence",
                         "🔄 Comparison", "🧠 Behavioral", "🤖 Model", "💬 Messages", "📚 Research"])
        (chat_tab, overview_tab, url_tab, sentiment_tab, emotion_tab, words_tab,
         conf_tab, comparison_tab, behavioral_tab, model_tab, all_messages_tab, research_tab) = tabs

        with chat_tab:
            uc = filtered_df["author"].value_counts()
            ma_user = uc.idxmax() if not uc.empty else "N/A"
            majority_sent = filtered_df[sentiment_col].mode()[0] if not filtered_df.empty else "N/A"
            avg_score = filtered_df["sentiment_score"].mean() if "sentiment_score" in filtered_df.columns else 0
            avg_sent = "Positive" if avg_score >= 0.05 else ("Negative" if avg_score <= -0.05 else "Neutral")
            overall_sent = avg_sent
            ec = "transformer_emotion" if "transformer_emotion" in filtered_df.columns else "emotion"
            top_em = filtered_df[ec].mode()[0] if ec in filtered_df.columns and not filtered_df.empty else "N/A"

            is_dark = st.session_state.get("theme","dark") == "dark"
            text_color = "#e6edf3" if is_dark else "#1a1a2e"
            muted = "#8b949e" if is_dark else "#6b7280"
            card_bg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
            border = "rgba(88,166,255,0.18)" if is_dark else "rgba(99,102,241,0.18)"

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            stats = [
                ("💬", "Total Messages", f"{len(filtered_df):,}", "#58a6ff"),
                ("👥", "Active Users", f"{filtered_df['author'].nunique():,}", "#bc8cff"),
                ("🏆", "Most Active", str(ma_user), "#fbbf24"),
                ("🎭", "Top Emotion", str(top_em).capitalize(), "#3fb950"),
                ("🗳️", "Majority Sentiment", majority_sent, "#f78166"),
                ("📊", "Avg Sentiment", "Positive" if filtered_df["sentiment_score"].mean() >= 0.05 else ("Negative" if filtered_df["sentiment_score"].mean() <= -0.05 else "Neutral"), "#bc8cff"),
            ]
            for col, (icon, label, val, accent) in zip([c1,c2,c3,c4,c5,c6], stats):
                with col:
                    st.markdown(f"""
                    <div style="background:{card_bg};border:1px solid {accent}30;
                                border-radius:14px;padding:18px 16px;
                                border-top:3px solid {accent};
                                box-shadow:0 4px 20px {accent}15;margin-bottom:8px;">
                        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:1.2px;color:{muted};margin-bottom:8px;">{icon} {label}</div>
                        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                                    font-weight:700;color:{accent};">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Info cards
            sent_color = "#3fb950" if str(overall_sent).lower()=="positive" else ("#f78166" if str(overall_sent).lower()=="negative" else "#58a6ff")
            st.markdown(f"""
            <div style="background:{card_bg};border:1px solid {border};border-radius:14px;
                        padding:16px 20px;margin:8px 0 16px 0;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;
                            padding-bottom:10px;border-bottom:1px solid {border};">
                    <span style="font-size:1rem;">📅</span>
                    <span style="color:{muted};font-size:0.83rem;">Date Range:</span>
                    <span style="color:{text_color};font-weight:600;font-size:0.83rem;">
                        {filtered_df['datetime'].min().strftime('%Y-%m-%d')}
                        &nbsp;→&nbsp;
                        {filtered_df['datetime'].max().strftime('%Y-%m-%d')}
                    </span>
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;
                            padding-bottom:10px;border-bottom:1px solid {border};">
                    <span style="font-size:1rem;">😊</span>
                    <span style="color:{muted};font-size:0.83rem;">Overall Sentiment:</span>
                    <span style="color:{sent_color};font-weight:700;font-size:0.85rem;">
                        {overall_sent.capitalize()} (Avg) · {majority_sent.capitalize()} (Majority)
                    </span>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:1rem;">🏆</span>
                    <span style="color:{muted};font-size:0.83rem;">Most Active User:</span>
                    <span style="color:#fbbf24;font-weight:700;font-size:0.85rem;">
                        {ma_user}
                    </span>
                    <span style="color:{muted};font-size:0.78rem;">({uc.max():,} messages)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Export button
            csv_sum = pd.DataFrame({
                "Metric": ["Total Messages","Active Users","Most Active","Sentiment","Top Emotion"],
                "Value": [len(filtered_df), filtered_df['author'].nunique(), ma_user, overall_sent, str(top_em).capitalize()]
            }).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Summary CSV", csv_sum, "chat_summary.csv", "text/csv", key="chat_summary_export")

            ui_divider()
            if not filtered_df.empty:
                summary_text = generate_text_summary(filtered_df)
                st.markdown(f"""
                <div style="background:{card_bg};padding:22px 26px;border-radius:16px;
                            border:1px solid {border};border-left:4px solid #58a6ff;
                            backdrop-filter:blur(10px);">
                    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:1.2px;color:#58a6ff;margin-bottom:12px;">
                        🤖 Chat Summary
                    </div>
                    <p style="color:{text_color};line-height:1.8;font-size:0.92rem;margin:0;">{summary_text}</p>
                </div>
                """, unsafe_allow_html=True)



        with overview_tab:
            c1,c2,c3,c4 = st.columns(4)
            with c1: ui_metric_card("Total Messages", f"{len(filtered_df):,}", "💬")
            with c2: ui_metric_card("Active Users", f"{filtered_df['author'].nunique():,}", "👥")
            with c3: ui_metric_card("Positive", f"{len(filtered_df[filtered_df[sentiment_col].str.lower()=='positive']):,}", "✅")
            with c4: ui_metric_card("Negative", f"{len(filtered_df[filtered_df[sentiment_col].str.lower()=='negative']):,}", "❌")
            st.plotly_chart(create_message_heatmap(filtered_df), use_container_width=True, key="trans_ov_heatmap")
            udf = filtered_df["author"].value_counts().reset_index()
            udf.columns = ["User","Messages"]
            c1,c2 = st.columns([1,2])
            with c1:
                is_dark = st.session_state.get("theme","dark") == "dark"
                text_color = "#e6edf3" if is_dark else "#1a1a2e"
                card_bg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
                border = "rgba(88,166,255,0.15)" if is_dark else "rgba(99,102,241,0.15)"
                muted = "#8b949e" if is_dark else "#6b7280"
                rows_html = ""
                for i, row in udf.iterrows():
                    row_bg = "rgba(88,166,255,0.04)" if i % 2 == 0 else "transparent"
                    rows_html += f"<tr style='background:{row_bg};'><td style='padding:10px 14px;color:{text_color};font-weight:500;font-size:0.85rem;border-bottom:1px solid {border};'>{row['User']}</td><td style='padding:10px 14px;color:#58a6ff;font-weight:700;font-size:0.9rem;border-bottom:1px solid {border};text-align:right;'>{row['Messages']}</td></tr>"
                st.markdown(f"""
                <div style="background:{card_bg};border:1px solid {border};border-radius:14px;overflow:hidden;backdrop-filter:blur(20px);">
                    <table style="width:100%;border-collapse:collapse;">
                        <thead><tr style="background:linear-gradient(135deg,#58a6ff,#bc8cff);">
                            <th style="padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">👤 User</th>
                            <th style="padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:right;">💬 Messages</th>
                        </tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>""", unsafe_allow_html=True)
            with c2:
                fig_user_bar = px.bar(udf, x="User", y="Messages", color="Messages",
                                      color_continuous_scale="Viridis")
                st.plotly_chart(apply_chart_theme(fig_user_bar,"👥 Messages per User"), use_container_width=True, key="trans_user_bar")

        with url_tab:
            url_data = []
            for _, row in filtered_df.iterrows():
                msg = str(row.get("message_raw", row.get("message", "")))
                for url in re.findall(r"(https?://\S+|www\.\S+)", msg):
                    url_data.append({"User": row["author"], "URL": url, "Date": row["datetime"].date()})
            if url_data:
                is_dark = st.session_state.get("theme","dark") == "dark"
                text_color = "#e6edf3" if is_dark else "#1a1a2e"
                card_bg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
                border = "rgba(88,166,255,0.15)" if is_dark else "rgba(99,102,241,0.15)"
                udf_url = pd.DataFrame(url_data)
                ui_metric_card("URLs Found", len(udf_url), "🔗")
                # Styled bar chart using plotly
                vc = udf_url["User"].value_counts().reset_index()
                vc.columns = ["User","Count"]
                fig_url = px.bar(vc, x="User", y="Count", color="Count", color_continuous_scale="Blues")
                st.plotly_chart(apply_chart_theme(fig_url, "🔗 URLs per User"), use_container_width=True, key="url_bar")
                rows_html = ""
                for i, r in udf_url.iterrows():
                    u = r["URL"]
                    href = u if u.startswith("http") else "https://" + u
                    row_bg = "rgba(88,166,255,0.04)" if i % 2 == 0 else "transparent"
                    rows_html += f"<tr style='background:{row_bg};'><td style='padding:10px 14px;color:{text_color};font-size:0.83rem;border-bottom:1px solid {border};'>{r['User']}</td><td style='padding:10px 14px;border-bottom:1px solid {border};'><a href='{href}' target='_blank' style='color:#58a6ff;font-size:0.83rem;'>{u}</a></td><td style='padding:10px 14px;color:#8b949e;font-size:0.8rem;border-bottom:1px solid {border};'>{r['Date']}</td></tr>"
                st.markdown(f"""
                <div style="background:{card_bg};border:1px solid {border};border-radius:14px;overflow:hidden;backdrop-filter:blur(20px);margin-top:16px;">
                    <table style="width:100%;border-collapse:collapse;">
                        <thead><tr style="background:linear-gradient(135deg,#58a6ff,#bc8cff);">
                            <th style="padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">👤 User</th>
                            <th style="padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">🔗 URL</th>
                            <th style="padding:11px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">📅 Date</th>
                        </tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("No URLs found.")

        with sentiment_tab:
            render_sentiment_section(filtered_df, sentiment_col, key_prefix="trans")

        with emotion_tab:
            ec = "transformer_emotion" if "transformer_emotion" in filtered_df.columns else "emotion"
            if ec in filtered_df.columns and not filtered_df[ec].isna().all():
                render_emotion_section(filtered_df, key_prefix="trans")
            else:
                is_dark = st.session_state.get("theme","dark") == "dark"
                tc = "#e6edf3" if is_dark else "#1a1a2e"
                st.markdown(f"""
                <div style="text-align:center;padding:40px;color:{tc};">
                    <div style="font-size:3rem;">🎭</div>
                    <div style="font-size:1.2rem;font-weight:600;margin-top:12px;">
                        Enable Emotion Detection
                    </div>
                    <div style="font-size:0.9rem;color:#8b949e;margin-top:8px;">
                        Check "Include Emotion Detection" in sidebar and re-analyze
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with words_tab:
            ec = "transformer_emotion" if "transformer_emotion" in filtered_df.columns else "emotion"
            render_top_emotional_words(filtered_df, ec)

        with conf_tab:
            st.plotly_chart(create_confidence_score_chart(filtered_df), use_container_width=True, key="conf_chart")

        with comparison_tab:
            st.plotly_chart(create_sentiment_emotion_comparison(filtered_df), use_container_width=True, key="sent_em_comp")

        with behavioral_tab:
            st.markdown("### ☠️ Toxicity Analysis")
            um = filtered_df.groupby("author").size()
            neg_mask = filtered_df[sentiment_col].str.lower() == "negative"
            nm = filtered_df[neg_mask].groupby("author").size().reindex(um.index, fill_value=0)
            scores = (nm / um * 100).fillna(0).round(1)
            ot = (neg_mask.sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            color = "#3fb950" if ot <= 30 else ("#f78166" if ot > 60 else "#fbbf24")
            level = "Healthy 🟢" if ot <= 30 else ("Moderate 🟡" if ot <= 60 else "Toxic 🔴")
            ui_metric_card("Group Toxicity", f"{ot:.1f}%", "☠️")
            st.markdown(f"**Level:** <span style='color:{color};font-weight:700;'>{level}</span>", unsafe_allow_html=True)
            ui_divider()
            if not scores.empty:
                fig = px.bar(scores, x=scores.index, y=scores.values, color=scores.values,
                              color_continuous_scale="Reds",
                              labels={"y": "Toxicity %", "x": "User"})
                st.plotly_chart(apply_chart_theme(fig, "Toxicity by User"), use_container_width=True, key="toxicity_bar")
            ui_divider()
            if filtered_df["author"].nunique() >= 2:
                us = pd.crosstab(filtered_df["author"], filtered_df[sentiment_col], normalize="index").mul(100).round(1)
                # Normalize columns to lowercase
                us.columns = [str(c).lower() for c in us.columns]
                for col in ["positive","negative","neutral"]:
                    if col not in us.columns: us[col] = 0
                pos_col = "positive"
                neg_col = "negative"
                most_pos = us[pos_col].idxmax()
                most_neg = us[neg_col].idxmax()
                pos_pct = us[pos_col].max()
                neg_pct = us[neg_col].max()
                is_dark = st.session_state.get("theme","dark") == "dark"
                tc = "#e6edf3" if is_dark else "#1a1a2e"
                cbg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
                st.markdown(f"""
                <div style="display:flex;gap:16px;margin-top:16px;">
                    <div style="flex:1;background:{cbg};border:1px solid #3fb95030;
                                border-top:3px solid #3fb950;border-radius:14px;padding:20px;">
                        <div style="font-size:0.72rem;color:#3fb950;font-weight:700;
                                    text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                            😊 Most Positive User
                        </div>
                        <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
                                    font-weight:700;color:#3fb950;">{most_pos}</div>
                        <div style="font-size:0.85rem;color:#8b949e;margin-top:4px;">
                            {pos_pct:.1f}% positive messages
                        </div>
                    </div>
                    <div style="flex:1;background:{cbg};border:1px solid #f7816630;
                                border-top:3px solid #f78166;border-radius:14px;padding:20px;">
                        <div style="font-size:0.72rem;color:#f78166;font-weight:700;
                                    text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                            😠 Most Negative User
                        </div>
                        <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
                                    font-weight:700;color:#f78166;">{most_neg}</div>
                        <div style="font-size:0.85rem;color:#8b949e;margin-top:4px;">
                            {neg_pct:.1f}% negative messages
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with model_tab:
            render_model_performance_tab(filtered_df, analysis_method, selected_language,
                                          enable_lang_detection, enable_emoji_analysis,
                                          enable_multimodal_fusion, sentiment_col)

        with research_tab:
            render_research_tab()

        with all_messages_tab:
            sentiment_filter_options = ["All"] + sorted(filtered_df[sentiment_col].unique().tolist())
            sel_sent = st.selectbox("Filter by Sentiment", sentiment_filter_options, key="trans_sent_filter")
            temp_df = filtered_df.copy()
            if sel_sent != "All":
                temp_df = temp_df[temp_df[sentiment_col] == sel_sent]
            c1,c2,c3 = st.columns(3)
            with c1: lim = st.number_input("Show", 10, 1000, 50, 10, key="trans_lim")
            with c2: sb = st.selectbox("Sort", ["datetime", sentiment_col, "sentiment_score"], key="trans_sort")
            with c3: asc = st.checkbox("Ascending", False, key="trans_asc")
            dcols = [c for c in ["datetime","author","message",sentiment_col,"sentiment_score"] if c in temp_df.columns]
            display_df = temp_df[dcols].sort_values(sb, ascending=asc).head(int(lim)).copy()
            display_df["datetime"] = display_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")

            is_dark = st.session_state.get("theme","dark") == "dark"
            text_color = "#e6edf3" if is_dark else "#1a1a2e"
            card_bg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
            border = "rgba(88,166,255,0.15)" if is_dark else "rgba(99,102,241,0.15)"
            muted = "#8b949e" if is_dark else "#6b7280"

            # Always use solid colors that work in both themes
            SENT_COLORS = {
                "positive": ("#166534", "#ffffff"),
                "negative": ("#7f1d1d", "#ffffff"),
                "neutral":  ("#1e3a5f", "#ffffff"),
            }

            rows_html = ""
            for _, row in display_df.iterrows():
                sent = str(row.get(sentiment_col,"neutral")).lower()
                bg_s, fg_s = SENT_COLORS.get(sent, ("#1a2744","#58a6ff"))
                score = f"{row['sentiment_score']:.2f}" if "sentiment_score" in row else ""
                msg = str(row.get("message",""))[:80] + ("..." if len(str(row.get("message",""))) > 80 else "")
                rows_html += f"""
                <tr style="border-bottom:1px solid {border};transition:background 0.2s;">
                    <td style="padding:10px 14px;color:{muted};font-size:0.78rem;white-space:nowrap;">{row['datetime']}</td>
                    <td style="padding:10px 14px;color:{text_color};font-size:0.82rem;font-weight:600;">{row.get('author','')}</td>
                    <td style="padding:10px 14px;color:{text_color};font-size:0.82rem;max-width:300px;">{msg}</td>
                    <td style="padding:10px 14px;">
                        <span style="background:{bg_s};color:{fg_s};padding:3px 10px;
                                     border-radius:20px;font-size:0.75rem;font-weight:700;
                                     border:1px solid {fg_s}40;">{sent.capitalize()}</span>
                    </td>
                    <td style="padding:10px 14px;color:{text_color};font-size:0.82rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{score}</td>
                </tr>"""

            st.markdown(f"""
            <div style="background:{card_bg};border:1px solid {border};border-radius:16px;
                        overflow:hidden;backdrop-filter:blur(20px);margin-top:16px;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead>
                        <tr style="background:linear-gradient(135deg,#58a6ff,#bc8cff);">
                            <th style="padding:12px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">📅 Date</th>
                            <th style="padding:12px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">👤 Author</th>
                            <th style="padding:12px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">💬 Message</th>
                            <th style="padding:12px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">😊 Sentiment</th>
                            <th style="padding:12px 14px;color:white;font-size:0.78rem;font-weight:700;text-align:left;">🎯 Score</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            st.download_button("📥 Download CSV", display_df.to_csv(index=False),
                               "messages.csv", "text/csv", key="trans_dl")

    # ── Multimodal Dashboard ──
    elif "Multimodal" in analysis_method:
        st.markdown("<h1 style='text-align:center;'>🌐 Multimodal Dashboard</h1>", unsafe_allow_html=True)
        (ov_tab, emoji_tab, fusion_tab, sent_tab, em_tab, users_tab, words_tab,
         act_tab, top_tab, all_tab, model_tab, research_tab) = st.tabs([
            "📈 Overview", "🎭 Emoji Analysis", "🔀 Text+Emoji Fusion",
            "😊 Sentiment", "😊 Emotions", "👥 Users",
            "☁️ Words", "📅 Activity", "🏆 Top", "💬 Messages", "🤖 Model", "📚 Research"
        ])
        # Shared emoji computation
        import emoji as emoji_lib
        all_emojis = []
        for msg in filtered_df["message"].astype(str):
            for e in emoji_lib.emoji_list(msg):
                all_emojis.append(e["emoji"])
        pos_emojis = set(["😊","😂","😍","👍","🎉","❤️","🔥","💪","✅","🙌","😁","💯","😄"])
        neg_emojis = set(["😠","😢","👎","💔","😡","😭","🤦","😤","💀","🙄","😒","😞","😔","😣"])

        with ov_tab:
            c1,c2,c3,c4 = st.columns(4)
            with c1: ui_metric_card("Total Messages", f"{len(filtered_df):,}", "💬")
            with c2: ui_metric_card("Positive", f"{len(filtered_df[filtered_df[sentiment_col].str.lower()=='positive']):,}", "✅")
            with c3: ui_metric_card("Negative", f"{len(filtered_df[filtered_df[sentiment_col].str.lower()=='negative']):,}", "❌")
            with c4: ui_metric_card("Active Users", f"{filtered_df['author'].nunique():,}", "👥")
            ui_divider()
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(create_sentiment_pie_chart(filtered_df, "sentiment"), use_container_width=True, key="mm_pie")
            with c2: st.plotly_chart(create_message_heatmap(filtered_df), use_container_width=True, key="mm_heatmap")

        with emoji_tab:
            st.markdown("### 🎭 Emoji Analysis")
            ec1,ec2,ec3 = st.columns(3)
            with ec1: ui_metric_card("Total Emojis", len(all_emojis), "🎭")
            with ec2: ui_metric_card("Positive Emojis", sum(1 for e in all_emojis if e in pos_emojis), "😊")
            with ec3: ui_metric_card("Negative Emojis", sum(1 for e in all_emojis if e in neg_emojis), "😢")
            if all_emojis:
                ui_divider()
                emoji_counts = Counter(all_emojis).most_common(15)
                emojis_df = pd.DataFrame(emoji_counts, columns=["Emoji","Count"])
                fig_emoji = go.Figure(data=[go.Bar(
                    x=emojis_df["Emoji"], y=emojis_df["Count"],
                    marker=dict(color=emojis_df["Count"],colorscale="Viridis"))])
                st.plotly_chart(apply_chart_theme(fig_emoji,"🎭 Top Emojis Used"),
                                use_container_width=True, key="mm_emoji_bar")
                ui_divider()
                pos_e = sum(1 for e in all_emojis if e in pos_emojis)
                neg_e = sum(1 for e in all_emojis if e in neg_emojis)
                neu_e = len(all_emojis) - pos_e - neg_e
                fig_epie = go.Figure(data=[go.Pie(
                    labels=["Positive 😊","Negative 😢","Neutral 😐"],
                    values=[pos_e, neg_e, neu_e], hole=0.45,
                    marker=dict(colors=["#3fb950","#f78166","#58a6ff"]))])
                st.plotly_chart(apply_chart_theme(fig_epie,"😊 Emoji Sentiment Distribution"),
                                use_container_width=True, key="mm_emoji_pie")
            else:
                st.info("No emojis found in this chat.")
            ui_divider()
            # Per User Emoji Analysis
            st.markdown("### 👤 Emoji Usage by User")
            user_emoji_data = []
            for _, row in filtered_df.iterrows():
                msg_emojis = [e["emoji"] for e in emoji_lib.emoji_list(str(row["message"]))]
                for em in msg_emojis:
                    sentiment = "Positive 😊" if em in pos_emojis else ("Negative 😢" if em in neg_emojis else "Neutral 😐")
                    user_emoji_data.append({"User": row["author"], "Emoji": em, "Sentiment": sentiment})
            if user_emoji_data:
                uem_df = pd.DataFrame(user_emoji_data)
                # Per user emoji count
                user_counts = uem_df.groupby(["User","Sentiment"]).size().unstack(fill_value=0).reset_index()
                fig_user_emoji = go.Figure()
                colors = {"Positive 😊":"#3fb950","Negative 😢":"#f78166","Neutral 😐":"#58a6ff"}
                for col in [c for c in user_counts.columns if c != "User"]:
                    fig_user_emoji.add_trace(go.Bar(
                        name=col, x=user_counts["User"],
                        y=user_counts[col],
                        marker_color=colors.get(col,"#8b949e")))
                fig_user_emoji.update_layout(barmode="group")
                st.plotly_chart(apply_chart_theme(fig_user_emoji,
                    "👤 Emoji Sentiment by User"),
                    use_container_width=True, key="mm_user_emoji_bar")
                ui_divider()
                # Top emoji per user table
                is_dark = st.session_state.get("theme","dark") == "dark"
                tc = "#e6edf3" if is_dark else "#1a1a2e"
                cbg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
                br = "rgba(88,166,255,0.15)" if is_dark else "rgba(99,102,241,0.15)"
                st.markdown(f"<div style='background:{cbg};border:1px solid {br};border-radius:14px;overflow:hidden;'>", unsafe_allow_html=True)
                cols = st.columns(len(filtered_df["author"].unique()))
                for col, user in zip(cols, sorted(filtered_df["author"].unique())):
                    with col:
                        user_emojis = uem_df[uem_df["User"]==user]["Emoji"].value_counts().head(5)
                        if not user_emojis.empty:
                            rows_html = ""
                            for em, cnt in user_emojis.items():
                                sent = "😊" if em in pos_emojis else ("😢" if em in neg_emojis else "😐")
                                rows_html += f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {br};'><span style='font-size:1.2rem;'>{em}</span><span style='color:#fbbf24;font-weight:700;'>{cnt}x</span><span>{sent}</span></div>"
                            st.markdown(f"""
                            <div style="background:{cbg};border:1px solid {br};
                                        border-radius:12px;padding:16px;margin:4px;">
                                <div style="color:#58a6ff;font-weight:700;font-size:0.8rem;
                                            margin-bottom:10px;text-transform:uppercase;">
                                    👤 {user}
                                </div>
                                {rows_html}
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with fusion_tab:
            st.markdown("### 🔀 Text + Emoji Fusion Analysis")
            if "mm_text_sentiment" in filtered_df.columns and "mm_emoji_sentiment" in filtered_df.columns:
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    avg_text = filtered_df["mm_text_sentiment"].mean()
                    ui_metric_card("Avg Text Sentiment", f"{avg_text:.3f}", "📝")
                with fc2:
                    avg_emoji = filtered_df["mm_emoji_sentiment"].mean()
                    ui_metric_card("Avg Emoji Sentiment", f"{avg_emoji:.3f}", "🎭")
                with fc3:
                    avg_combined = (avg_text + avg_emoji) / 2
                    ui_metric_card("Avg Fused Sentiment", f"{avg_combined:.3f}", "🔀")
                ui_divider()
                fusion_df = filtered_df[["author","mm_text_sentiment","mm_emoji_sentiment"]].groupby("author").mean().reset_index()
                fig_fusion = go.Figure()
                fig_fusion.add_trace(go.Bar(name="📝 Text Sentiment",
                    x=fusion_df["author"], y=fusion_df["mm_text_sentiment"],
                    marker_color="#58a6ff"))
                fig_fusion.add_trace(go.Bar(name="🎭 Emoji Sentiment",
                    x=fusion_df["author"], y=fusion_df["mm_emoji_sentiment"],
                    marker_color="#bc8cff"))
                fig_fusion.update_layout(barmode="group")
                st.plotly_chart(apply_chart_theme(fig_fusion,"🔀 Text vs Emoji Sentiment by User"),
                                use_container_width=True, key="mm_fusion_bar")
                ui_divider()
                filtered_df_copy = filtered_df.copy()
                filtered_df_copy["text_label"] = filtered_df_copy["mm_text_sentiment"].apply(
                    lambda x: "Positive" if x>0.1 else ("Negative" if x<-0.1 else "Neutral"))
                filtered_df_copy["emoji_label"] = filtered_df_copy["mm_emoji_sentiment"].apply(
                    lambda x: "Positive" if x>0.1 else ("Negative" if x<-0.1 else "Neutral"))
                filtered_df_copy["agree"] = filtered_df_copy["text_label"]==filtered_df_copy["emoji_label"]
                agree_pct = filtered_df_copy["agree"].mean()*100
                fa1, fa2 = st.columns(2)
                with fa1: ui_metric_card("Text-Emoji Agreement", f"{agree_pct:.1f}%", "✅")
                with fa2: ui_metric_card("Text-Emoji Disagreement", f"{100-agree_pct:.1f}%", "⚡")
                ui_divider()
                # Show messages where text and emoji sentiment disagreed
                is_dark = st.session_state.get("theme","dark") == "dark"
                tc = "#e6edf3" if is_dark else "#1a1a2e"
                cbg = "rgba(22,27,34,0.95)" if is_dark else "rgba(255,255,255,0.95)"
                br = "rgba(88,166,255,0.15)" if is_dark else "rgba(99,102,241,0.15)"
                import emoji as emoji_lib
                filtered_df_copy["has_emoji"] = filtered_df_copy["message"].astype(str).apply(
                    lambda x: len(emoji_lib.emoji_list(x)) > 0)
                disagree_df = filtered_df_copy[
                    (~filtered_df_copy["agree"]) & 
                    (filtered_df_copy["has_emoji"])
                ].head(10)
                if disagree_df.empty:
                    disagree_df = filtered_df_copy[~filtered_df_copy["agree"]].head(10)
                if not disagree_df.empty:
                    st.markdown(f"""
                    <div style="background:{cbg};border:1px solid #f7816630;
                                border-top:3px solid #f78166;border-radius:14px;
                                padding:16px 20px;margin-bottom:16px;">
                        <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:1.2px;color:#f78166;margin-bottom:8px;">
                            ⚡ Messages Where Text & Emoji Sentiment Disagreed
                        </div>
                        <div style="font-size:0.82rem;color:#8b949e;">
                            📌 These messages contain emojis where emoji sentiment 
                            differs from text sentiment — proving why multimodal 
                            fusion gives better results than text-only analysis
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    rows_html = ""
                    for i, row in disagree_df.iterrows():
                        rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                        txt_color = "#3fb950" if row["text_label"]=="Positive" else ("#f78166" if row["text_label"]=="Negative" else "#58a6ff")
                        em_color = "#3fb950" if row["emoji_label"]=="Positive" else ("#f78166" if row["emoji_label"]=="Negative" else "#58a6ff")
                        msg = str(row.get("message",""))[:60] + "..." if len(str(row.get("message","")))>60 else str(row.get("message",""))
                        rows_html += f"""<tr style='background:{rbg};'>
                            <td style='padding:8px 12px;color:{tc};font-size:0.8rem;border-bottom:1px solid {br};'>{row['author']}</td>
                            <td style='padding:8px 12px;color:{tc};font-size:0.78rem;border-bottom:1px solid {br};'>{msg}</td>
                            <td style='padding:8px 12px;border-bottom:1px solid {br};'><span style='color:{txt_color};font-weight:700;font-size:0.78rem;'>{row['text_label']}</span></td>
                            <td style='padding:8px 12px;border-bottom:1px solid {br};'><span style='color:{em_color};font-weight:700;font-size:0.78rem;'>{row['emoji_label']}</span></td>
                        </tr>"""
                    st.markdown(f"""
                    <div style="background:{cbg};border:1px solid {br};border-radius:14px;overflow:hidden;">
                        <table style="width:100%;border-collapse:collapse;">
                            <thead><tr style="background:linear-gradient(135deg,#f78166,#bc8cff);">
                                <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">👤 User</th>
                                <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">💬 Message</th>
                                <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">📝 Text</th>
                                <th style="padding:10px 12px;color:white;font-size:0.75rem;text-align:left;">🎭 Emoji</th>
                            </tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Enable Multimodal Fusion in sidebar settings.")
        with sent_tab:
            render_sentiment_section(filtered_df, sentiment_col, key_prefix="mm")
        with em_tab:
            render_emotion_section(filtered_df, key_prefix="mm")
        with users_tab:
            if "author" in filtered_df.columns and not filtered_df.empty:
                su = st.selectbox("Select user", filtered_df["author"].unique().tolist(), key="mm_user_select")
                udf = filtered_df[filtered_df["author"] == su]
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(create_user_activity_gauge(filtered_df, su), use_container_width=True, key=f"mm_gauge_{su}")
                with c2: ui_metric_card("Messages", len(udf), "💬")
        with words_tab:
            c1,c2 = st.columns(2)
            with c1:
                if "message_raw" in filtered_df.columns:
                    wc_text = " ".join(filtered_df.apply(clean_text_for_wordcloud, axis=1).tolist())
                    if wc_text.strip():
                        save_wordcloud(wc_text, "wc.png")
                        st.image("wc.png", use_container_width=True)
            with c2:
                st.plotly_chart(create_word_frequency_chart(filtered_df, 15), use_container_width=True, key="mm_words")
        with act_tab:
            st.plotly_chart(create_message_heatmap(filtered_df), use_container_width=True, key="mm_act")
        with top_tab:
            if "sentiment_score" in filtered_df.columns:
                _is_dark = st.session_state.get("theme","dark") == "dark"
                _tc = "#e6edf3" if _is_dark else "#1a1a2e"
                _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
                _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
                _fdf = filtered_df.copy()
                _fdf["signed_score"] = _fdf.apply(
                    lambda row: row["sentiment_score"]
                    if str(row["sentiment"]).lower() == "positive"
                    else (-row["sentiment_score"]
                          if str(row["sentiment"]).lower() == "negative"
                          else 0.0), axis=1
                )
                c1,c2 = st.columns(2)
                for col, title, asc_val, hdr_color in [
                    (c1,"👍 Top Positive",False,"#3fb950"),
                    (c2,"👎 Top Negative",True,"#f78166")
                ]:
                    with col:
                        st.markdown(f"<h5 style='color:{_tc};margin-bottom:12px;'>{title}</h5>", unsafe_allow_html=True)
                        _tdf = _fdf.sort_values("signed_score", ascending=asc_val).head(5)[["datetime","author","message","signed_score"]]
                        _rows_t = ""
                        for i, r in _tdf.iterrows():
                            _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                            sc = "#3fb950" if float(r["signed_score"])>0 else ("#f78166" if float(r["signed_score"])<0 else "#8b949e")
                            _rows_t += f"<tr style='background:{_rbg};'><td style='padding:8px;color:#8b949e;font-size:0.75rem;border-bottom:1px solid {_br};'>{str(r['datetime'])[:16]}</td><td style='padding:8px;color:{_tc};font-weight:600;font-size:0.8rem;border-bottom:1px solid {_br};'>{r['author']}</td><td style='padding:8px;color:{_tc};font-size:0.78rem;border-bottom:1px solid {_br};'>{str(r['message'])[:50]}...</td><td style='padding:8px;color:{sc};font-weight:700;font-size:0.82rem;border-bottom:1px solid {_br};'>{r['signed_score']:.3f}</td></tr>"
                        st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:12px;overflow:hidden;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,{hdr_color},{hdr_color}99);'><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Date</th><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Author</th><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Message</th><th style='padding:9px 8px;color:white;font-size:0.72rem;'>Score</th></tr></thead><tbody>{_rows_t}</tbody></table></div>", unsafe_allow_html=True)
        with all_tab:
            _is_dark = st.session_state.get("theme","dark") == "dark"
            _tc = "#e6edf3" if _is_dark else "#1a1a2e"
            _bg = "rgba(22,27,34,0.95)" if _is_dark else "rgba(255,255,255,0.95)"
            _br = "rgba(88,166,255,0.15)" if _is_dark else "rgba(99,102,241,0.15)"
            c1,c2,c3 = st.columns(3)
            with c1: lim = st.number_input("Show", 10, 1000, 100, 10, key="mm_lim")
            with c2: sb = st.selectbox("Sort", [s for s in [sentiment_col,"datetime","author"] if s in filtered_df.columns], key="mm_sort")
            with c3: so = st.selectbox("Order", ["Descending","Ascending"], key="mm_order")
            ddf = filtered_df[["datetime","author","message",sentiment_col]].copy()
            ddf["datetime"] = ddf["datetime"].dt.strftime("%Y-%m-%d %H:%M")
            ddf = ddf.sort_values(sb, ascending=(so=="Ascending")).head(lim)
            SENT_COLORS_MM = {"positive":("#166534","#ffffff"),"negative":("#7f1d1d","#ffffff"),"neutral":("#1e3a5f","#ffffff")}
            _hd_mm = "".join([f"<th style='padding:10px 12px;color:white;font-size:0.75rem;font-weight:700;text-align:left;'>{c}</th>" for c in ddf.columns])
            _rows_mm = ""
            for i, r in ddf.iterrows():
                _rbg = "rgba(88,166,255,0.04)" if i%2==0 else "transparent"
                cells = ""
                for c, v in zip(ddf.columns, r.values):
                    if c == sentiment_col:
                        sbg, sfg = SENT_COLORS_MM.get(str(v).lower(), ("#1e3a5f","#ffffff"))
                        cells += f"<td style='padding:8px 12px;border-bottom:1px solid {_br};'><span style='background:{sbg};color:{sfg};padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;'>{v}</span></td>"
                    else:
                        cells += f"<td style='padding:8px 12px;color:{_tc};font-size:0.8rem;border-bottom:1px solid {_br};'>{str(v)[:60] if len(str(v))>60 else v}</td>"
                _rows_mm += f"<tr style='background:{_rbg};'>{cells}</tr>"
            st.markdown(f"<div style='background:{_bg};border:1px solid {_br};border-radius:14px;overflow:hidden;overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:linear-gradient(135deg,#58a6ff,#bc8cff);'>{_hd_mm}</tr></thead><tbody>{_rows_mm}</tbody></table></div>", unsafe_allow_html=True)
            st.download_button("📥 Download CSV", ddf.to_csv(index=False), "filtered_data.csv", "text/csv", key="mm_dl")
        with research_tab:
            render_research_tab()

        with model_tab:
            render_model_performance_tab(filtered_df, analysis_method, selected_language,
                                          enable_lang_detection, enable_emoji_analysis,
                                          enable_multimodal_fusion, sentiment_col)
    # ── Live Prediction ──
    ui_divider()
    st.markdown("### 🔮 Live Sentiment Predictor")
    text_input = st.text_input("Type any message to analyze:", key="live_pred_input",
                                placeholder="e.g. I love this project!")
    if text_input:
        sl, ss, el = live_prediction(text_input)
        bc = "#0d2818" if sl.lower()=="positive" else ("#2a0d0d" if sl.lower()=="negative" else "#0d1a2e")
        tc = "#3fb950" if sl.lower()=="positive" else ("#f78166" if sl.lower()=="negative" else "#58a6ff")
        ei = "😊" if sl.lower()=="positive" else ("😡" if sl.lower()=="negative" else "😐")
        pct = int(ss * 100)
        bar_w = pct
        st.markdown(f"""
        <div style='background:{bc};color:{tc};padding:24px 32px;border-radius:16px;
                    font-size:1.6rem;font-weight:700;text-align:center;
                    border:1px solid {tc}50;margin-bottom:16px;
                    font-family:"Playfair Display",serif;
                    box-shadow:0 8px 32px {tc}25;'>
            {sl} {ei}
        </div>
        <div style='background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.10);
                    border-radius:14px;padding:20px 24px;margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <span style='color:#e6edf3;font-size:0.95rem;font-weight:600;'>🎯 Confidence Score</span>
                <span style='color:{tc};font-size:1.6rem;font-weight:700;
                             font-family:"Playfair Display",serif;'>{pct}%</span>
            </div>
            <div style='height:12px;border-radius:6px;background:rgba(255,255,255,0.08);overflow:hidden;'>
                <div style='width:{bar_w}%;height:100%;border-radius:6px;
                            background:linear-gradient(90deg,{tc},{tc}88);
                            transition:width 0.5s ease;'></div>
            </div>
            <div style='display:flex;justify-content:space-between;margin-top:6px;'>
                <span style='color:#8b949e;font-size:0.72rem;'>0%</span>
                <span style='color:#8b949e;font-size:0.72rem;'>50%</span>
                <span style='color:#8b949e;font-size:0.72rem;'>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.get("include_emotion", False):
            st.info(f"🎭 Detected Emotion: **{el}**")

    ui_divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Export Results CSV",
                           filtered_df.to_csv(index=False).encode("utf-8"),
                           "sentiment_results.csv", "text/csv", key="export_main")
    with c2:
        if st.button("📄 Generate PDF Report"):
            pdf_data = generate_pdf_report(filtered_df, sentiment_col)
            st.download_button("📄 Download PDF", pdf_data, "sentiment_report.pdf",
                               "application/pdf", key="pdf_dl_btn")

# ════════════════════════════════════════════
#               LANDING PAGE
# ════════════════════════════════════════════
else:
    st.markdown("""
    <div style="text-align:center; padding:70px 20px 40px 20px;">
        <div style="font-family:'Playfair Display',serif; font-size:3.5rem;
                    font-weight:900; line-height:1.1; margin-bottom:18px;
                    background:linear-gradient(135deg,#58a6ff 0%,#bc8cff 50%,#f78166 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">
            WhatsApp<br>Sentiment Analyzer
        </div>
        <p style="font-size:1.15rem; color:var(--text-muted);
                  max-width:540px; margin:0 auto 36px auto; line-height:1.7;">
            AI-powered sentiment & emotion analysis for WhatsApp conversations.
            Uncover the mood behind every message.
        </p>
        <div style="background:linear-gradient(135deg,rgba(88,166,255,0.08),rgba(188,140,255,0.08));
                    border:1px solid rgba(88,166,255,0.25); border-radius:16px;
                    padding:18px 36px; display:inline-block;
                    animation:pulse 2.5s ease-in-out infinite;
                    box-shadow:0 0 30px rgba(88,166,255,0.15);">
            <span style="color:#58a6ff; font-weight:600; font-size:1rem;">
                🚀 Upload your WhatsApp export file in the sidebar to begin
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ui_divider()

    col1, col2, col3 = st.columns(3)
    cards = [
        ("📊", "Professional BI", "Power BI style visualizations — heatmaps, timelines, activity gauges, and user analytics for complete chat oversight."),
        ("🤖", "Advanced AI", "State-of-the-art NLP: VADER baseline + Transformer deep learning for precise sentiment and 7-class emotion detection."),
        ("🔍", "Behavioral Insights", "Toxicity scoring, sentiment shift detection, user profiling, and interaction pattern analysis.")
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; padding:30px 22px;">
                <div style="font-size:2.5rem; margin-bottom:14px;
                            filter:drop-shadow(0 0 12px rgba(88,166,255,0.3));">{icon}</div>
                <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                            font-weight:700; color:var(--text); margin-bottom:10px;">{title}</div>
                <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.65;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)