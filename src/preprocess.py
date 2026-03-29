import re
import pandas as pd
import emoji
from emoji import is_emoji
import string # Keep this import for string.punctuation

# Regex for extracting URLs
URL_REGEX = re.compile(r'https?://\S+')
URL_EXTRACT_REGEX = re.compile(r'(https?://[\w\-\.\?\,\'/\\\+&%\$#_=;~:()]+)')

# Common WhatsApp system messages and patterns to identify them
SYSTEM_MESSAGE_PATTERNS = [
    r'<Media omitted>',
    r'This message was deleted',
    r'Missed voice call',
    r'Missed video call',
    r'You created group ".*?"',
    r'.*? added .*?',
    r'.*? removed .*?',
    r'.*? left',
    r'.*? changed the group description',
    r'.*? changed to .*\.vcf', # Contact card messages
    r'Messages and calls are end-to-end encrypted\.' # E2E encryption message
]
SYSTEM_MESSAGE_REGEX = re.compile('|'.join(SYSTEM_MESSAGE_PATTERNS), re.IGNORECASE)

def is_system_message(text):
    """
    Checks if a message is a WhatsApp system message.
    """
    if not isinstance(text, str):
        return True # Treat non-string messages as system/unparseable
    return bool(SYSTEM_MESSAGE_REGEX.search(text.strip()))

def is_emoji_only_message(text):
    """
    Checks if a message consists entirely of emojis (after removing whitespace and non-emoji characters).
    """
    if not isinstance(text, str):
        return False
    
    # Remove all non-emoji characters and whitespace
    clean_text = ''.join(c for c in text if is_emoji(c) or c.isspace())
    clean_text = clean_text.strip()
    
    if not clean_text:
        return False
    
    # Check if all remaining characters are emojis
    return all(is_emoji(c) or c.isspace() for c in clean_text)


def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    # Remove URLs
    text = URL_REGEX.sub('', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove newlines and extra spaces
    text = ' '.join(text.split())
    
    return text


def extract_emojis(text):
    """
    Extract all emojis from a text string.
    """
    if not isinstance(text, str):
        return []
    return [match['emoji'] for match in emoji.emoji_list(text)]

def count_exclamations(text):
    """Count the number of exclamation marks in a string."""
    if not isinstance(text, str):
        return 0
    return text.count('!')

def count_questions(text):
    """Count the number of question marks in a string."""
    if not isinstance(text, str):
        return 0
    return text.count('?')

def count_capitalized_words(text):
    """Count the number of capitalized words in a string."""
    if not isinstance(text, str):
        return 0
    # A word is considered capitalized if it's fully uppercase and has > 1 char
    return sum(1 for word in text.split() if word.isupper() and len(word) > 1)

def extract_urls(text):
    if not isinstance(text, str):
        return []
    return URL_EXTRACT_REGEX.findall(text)


def preprocess_df(df, extract_emojis_flag=True):
    """Clean and prepare chat data, and engineer features for analysis.

    This function performs comprehensive preprocessing on the parsed WhatsApp chat
    DataFrame. It cleans text, extracts features, and prepares the data for
    sentiment analysis and visualization.

    Operations performed:
        - Extracts URLs into a 'urls' column.
        - Cleans message text by removing URLs and newlines.
        - Extracts and counts emojis, punctuation, and capitalized words.
        - Calculates word count.
        - Handles missing data.
        - Identifies system messages and emoji-only messages.

    Args:
        df (pd.DataFrame): DataFrame from parse_chat.
        extract_emojis_flag (bool): If True, extract emojis and count them.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional feature columns.
    """
    df = df.copy()
    
    # Ensure 'message' column exists and handle missing values
    if 'message' not in df.columns:
        df['message'] = ''
    df['message_raw'] = df['message'].fillna('')

    # --- Identify message types ---
    df['is_system_message'] = df['message_raw'].apply(is_system_message)
    df['is_emoji_only'] = df['message_raw'].apply(is_emoji_only_message)

    # --- Feature Engineering ---
    # Extract URLs before cleaning the message text
    df['urls'] = df['message_raw'].apply(extract_urls)
    
    # Extract and count emojis
    if extract_emojis_flag:
        df['emojis'] = df['message_raw'].apply(extract_emojis)
        df['emoji_count'] = df['emojis'].apply(len)
    else:
        df['emojis'] = [[] for _ in range(len(df))]
        df['emoji_count'] = 0

    # Count punctuation and capitalized words
    df['exclamation_count'] = df['message_raw'].apply(count_exclamations)
    df['question_count'] = df['message_raw'].apply(count_questions)
    df['caps_count'] = df['message_raw'].apply(count_capitalized_words)
    
    # --- Text Cleaning ---
    # Clean the message text for analysis (removes URLs, newlines)
    df['message'] = df['message_raw'].apply(clean_text)
    
    # Calculate word count on the cleaned message
    df['word_count'] = df['message'].apply(lambda s: len(s.split()))

    return df
