import pandas as pd
import emoji
import re
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="Hinglish & URL Tester", layout="wide")

# --- HINGLISH PREPROCESSING FUNCTION ---
def preprocess_hinglish(text):
    """
    Translates common Hinglish words to English before sentiment analysis
    to improve transformer model accuracy.
    """
    hinglish_dict = {
        'accha': 'good', 'acha': 'good', 'bahut': 'very', 'bohot': 'very',
        'nahi': 'no', 'nhi': 'no', 'haan': 'yes', 'han': 'yes',
        'yrr': 'friend', 'yaar': 'friend', 'bhai': 'brother',
        'stressed': 'stressed', 'pareshan': 'worried', 'dukhi': 'sad',
        'khush': 'happy', 'gussa': 'angry', 'dar': 'fear',
        'mushkil': 'difficult', 'takleef': 'pain', 'pyaar': 'love',
        'nafrat': 'hate', 'bekar': 'useless', 'ganda': 'bad',
        'mast': 'great', 'zabardast': 'amazing', 'bakwaas': 'nonsense',
        'bura': 'bad', 'acha nahi': 'not good', 'thand': 'cold',
        'rhe': 'are', 'rha': 'is', 'kyu': 'why', 'kya': 'what'
    }
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    for hindi, english in hinglish_dict.items():
        # Use regex for whole word replacement to avoid partial matches
        text_lower = re.sub(r'\b' + re.escape(hindi) + r'\b', english, text_lower)
    return text_lower

# --- HINGLISH DETECTION FUNCTION ---
def detect_hinglish(text):
    """
    Detects if a text contains Hinglish (Hindi written in Roman script) 
    based on a list of common Hindi words.
    """
    hindi_words = ['hai', 'hain', 'nahi', 'kya', 'yrr', 'yaar', 'bhai', 'tha', 
                   'thi', 'ho', 'kr', 'kar', 'rha', 'rhi', 'bohot', 'bahut', 
                   'acha', 'accha', 'theek', 'bilkul', 'abhi', 'phir', 'matlab',
                   'kyun', 'kaise', 'agar', 'lekin', 'aur', 'par', 'toh', 'bhi',
                   'mujhe', 'mere', 'teri', 'uska', 'unka', 'sab', 'kuch', 'log']
    if not isinstance(text, str):
        return False, 0
    text_lower = text.lower()
    words = text_lower.split()
    hindi_count = sum(1 for word in words if word in hindi_words)
    is_hinglish = hindi_count >= 1
    return is_hinglish, hindi_count

# --- ROBUST URL EXTRACTION FUNCTION ---
def extract_urls_robust(text):
    """
    Extracts standard URLs (http, https, www.) and detects link-sharing emojis (🔗, 🌐, 📎).
    """
    if not isinstance(text, str):
        return [], []
        
    # Standard URL pattern (http, https, www)
    url_pattern = r'(https?://\S+|www\.\S+)'
    urls = re.findall(url_pattern, text)
    
    # Link-sharing emojis
    link_emojis = ['🔗', '🌐', '📎']
    found_emojis = [e for e in link_emojis if e in text]
    
    return urls, found_emojis

# --- LOAD SENTIMENT MODEL ---
@st.cache_resource
def load_sentiment_pipeline():
    # Using a common sentiment model
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# --- 1. SIMULATE REAL-WORLD WHATSAPP DATA ---
chat_data = {
    'author': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Eve', 'Frank', 'Alice', 'Bob', 'Charlie'],
    'message': [
        "Check this out: https://example.com/page", 
        "Visit www.google.com for info",            
        "I found a cool resource 🔗",               
        "Check the global site 🌐 or attachment 📎", 
        "No links here, just a smile 😊",           
        "Multi-link: https://streamlit.io and https://github.com",
        "Protocol-less: bit.ly/test",
        "Kya haal hai bhai? Sab theek?", # Hinglish
        "Bohot acha kaam kar rha hai tu yrr!", # Hinglish
        "Bekar bakwaas message tha" # Hinglish Negative
    ]
}
df = pd.DataFrame(chat_data)

# Create Tabs
tabs = st.tabs(["🔗 URL & Emoji Fixer", "🤖 Transformer Dashboard"])

# --- TAB 1: URL & EMOJI FIXER ---
with tabs[0]:
    st.title("🔗 Robust URL & Emoji Fix Tester")
    
    # APPLY THE ROBUST FIX
    url_data = []
    for _, row in df.iterrows():
        urls, found_emojis = extract_urls_robust(row['message'])
        if urls or found_emojis:
            url_data.append({
                'User': row['author'],
                'URLs': ", ".join(urls) if urls else "None",
                'Emojis': ", ".join(found_emojis) if found_emojis else "None",
                'Full Message': row['message']
            })

    url_df = pd.DataFrame(url_data)

    if not url_df.empty:
        st.success(f"✅ Found {len(url_df)} link-related messages.")
        
        st.subheader("Top URL Sharers")
        st.table(url_df['User'].value_counts())

        st.subheader("Detailed Link Log (Clickable)")
        
        # Build full HTML with proper JS link opening
        html_content = """
        <html>
        <body style="margin:0; padding:0;">
        <table style="width:100%; border-collapse:collapse; font-family:sans-serif; font-size:14px; text-align:left;">
        <tr style="background:#6c63ff; color:white;">
          <th style="padding:12px;">User</th>
          <th style="padding:12px;">URLs</th>
          <th style="padding:12px;">Emojis</th>
          <th style="padding:12px;">Full Message</th>
        </tr>
        """

        for _, row in url_df.iterrows():
            url_text = row['URLs']
            if url_text != "None":
                links_html = ""
                for u in url_text.split(", "):
                    # Ensure protocol for window.open if missing
                    full_u = u if u.startswith('http') else 'http://' + u
                    links_html += f'<a href="{full_u}" onclick="window.open(\'{full_u}\',\'_blank\'); return false;" style="color:#1a73e8; text-decoration:underline; cursor:pointer; margin-right:10px;">{u}</a>'
            else:
                links_html = "None"

            html_content += f"""
            <tr style="border-bottom:1px solid #ddd;">
              <td style="padding:12px; white-space:nowrap;">{row['User']}</td>
              <td style="padding:12px;">{links_html}</td>
              <td style="padding:12px; text-align:center;">{row['Emojis']}</td>
              <td style="padding:12px;">{row['Full Message']}</td>
            </tr>
            """

        html_content += "</table></body></html>"
        components.html(html_content, height=400, scrolling=True)
    else:
        st.warning("No URLs found in sample data.")

# --- TAB 2: TRANSFORMER DASHBOARD ---
with tabs[1]:
    st.markdown("<h1 style='text-align: center;'>📊 Advanced Transformer Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("### 🌐 Hinglish / Code-Switch Detection")
    
    # Process Hinglish Detection
    df['is_hinglish'], df['hindi_count'] = zip(*df['message'].apply(detect_hinglish))
    
    # 2. Metric showing: Total Hinglish messages count
    total_hinglish = df['is_hinglish'].sum()
    st.metric("Total Hinglish Messages", total_hinglish)
    
    # 4. Highlight box showing percentage
    pct_hinglish = (total_hinglish / len(df)) * 100
    st.success(f"💡 {pct_hinglish:.1f}% of messages contain Hinglish - addressing RoBERTa's Twitter-training limitation")
    
    # --- SENTIMENT ANALYSIS WITH PREPROCESSING ---
    sentiment_pipe = load_sentiment_pipeline()
    
    # Apply Hinglish Preprocessing
    df['preprocessed_message'] = df['message'].apply(preprocess_hinglish)
    
    # Run Sentiment Analysis
    with st.spinner("Analyzing Sentiment..."):
        sent_results = sentiment_pipe(df['preprocessed_message'].tolist())
        
        # Map RoBERTa labels (LABEL_0: Negative, LABEL_1: Neutral, LABEL_2: Positive)
        label_map = {"LABEL_0": "Negative 😡", "LABEL_1": "Neutral 😐", "LABEL_2": "Positive 😊"}
        df['Sentiment'] = [label_map.get(r['label'], r['label']) for r in sent_results]
        df['Confidence'] = [f"{r['score']:.2%}" for r in sent_results]

    # 3. Bar chart: Hinglish vs Pure English messages per user
    hinglish_stats = df.groupby(['author', 'is_hinglish']).size().unstack(fill_value=0).reset_index()
    if True not in hinglish_stats.columns: hinglish_stats[True] = 0
    if False not in hinglish_stats.columns: hinglish_stats[False] = 0
    hinglish_stats = hinglish_stats.rename(columns={True: 'Hinglish', False: 'Pure English'})
    
    fig = px.bar(hinglish_stats, x='author', y=['Hinglish', 'Pure English'], 
                 title="Hinglish vs Pure English Messages per User",
                 barmode='group',
                 color_discrete_map={'Hinglish': '#ff4b4b', 'Pure English': '#0068c9'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 1. New column in message table: "Hinglish" → Yes / No badge
    st.subheader("Message Analysis Log (with Hinglish Fix)")
    
    display_df = df.copy()
    display_df['Hinglish'] = display_df['is_hinglish'].apply(lambda x: "Yes ✅" if x else "No ❌")
    
    # Show the preprocessing effect
    st.dataframe(display_df[['author', 'message', 'preprocessed_message', 'Hinglish', 'Sentiment', 'Confidence']], use_container_width=True)
