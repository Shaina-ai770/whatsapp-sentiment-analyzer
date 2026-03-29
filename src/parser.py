import re
from dateutil import parser as dateparser
import pandas as pd
import io

# Minimal WhatsApp exported chat parser
# Supports formats like:
# "12/31/20, 9:12 PM - Alice: Hello"
# "31/12/2020, 21:12 - Alice: Hello" (different locales)

# Updated regex patterns for different WhatsApp chat formats
PATTERN1_REGEX = re.compile(r"^(?P<date>\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}),?\s*(?P<time>\d{1,2}:\d{2}(?:\s*[AP]M)?)\s*-\s*(?P<author_raw>[^:]+): (?P<message_raw>.+)$", re.IGNORECASE)
PATTERN2_REGEX = re.compile(r"^(?P<date>\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}),?\s*(?P<time>\d{1,2}:\d{2}:\d{2}(?:\s*[AP]M)?)\s*-\s*(?P<author_raw>[^:]+): (?P<message_raw>.+)$", re.IGNORECASE)
MOBILE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")

def parse_chat(chat_data, encoding='utf-8'):
    """Parse an exported WhatsApp chat file into a structured pandas DataFrame.

    This function reads exported WhatsApp chat text files and converts them into
    structured data format for analysis. It handles various date/time formats,
    multi-line messages, mobile numbers, and media references.

    Args:
        chat_data (str): The raw chat text content as a string, or file path
        encoding (str): Character encoding to use (default: 'utf-8')

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - datetime: Parsed datetime object for each message
            - author: Name of the message sender (None for system messages)
            - mobile: Extracted mobile number if present (None otherwise)
            - media: Detected media files or 'media_omitted' placeholder
            - message: The actual message text content

    Supported WhatsApp formats:
        - "12/31/20, 9:12 PM - Alice: Hello" (flexible date/time, no seconds)
        - "12/03/2024, 09:02:11 - Arjun Kapoor: Good morning" (flexible date/time, with seconds)
        - Multi-line messages are automatically concatenated
        - System messages without authors are supported

    Example:
        >>> chat_content = "12/31/20, 9:12 PM - Alice: Hello\\n12/31/20, 9:13 PM - Bob: Hi!"
        >>> df = parse_chat(chat_content)
        >>> print(df.columns)
        Index(['datetime', 'author', 'mobile', 'media', 'message'], dtype='object')
    """
    rows = []
    f = io.StringIO(chat_data)
    cur = None
    for line in f:
        line = line.strip('\n')
        
        m = PATTERN1_REGEX.match(line)
        if not m:
            m = PATTERN2_REGEX.match(line)
            
        if m:
            date_str = m.group('date')
            time_str = m.group('time')
            author_raw = m.group('author_raw')
            message = m.group('message_raw')

            mobile = None
            mnum = MOBILE_REGEX.search(author_raw)
            if mnum:
                mobile = mnum.group(1).strip()
                author = MOBILE_REGEX.sub('', author_raw).strip()
                author = re.sub(r"^[\(\s\)]+|[\(\s\)]+$", '', author)
            else:
                author = author_raw

            if author == '':
                author = None

            media = None
            if '<Media omitted>' in message or 'image omitted' in message.lower():
                media = 'media_omitted'
            else:
                mfile = re.search(r"[\w-]{3,}\.((jpg|jpeg|png|mp4|mov|pdf|gif|webp))", message, flags=re.IGNORECASE)
                if mfile:
                    media = mfile.group(0)
            
            dt = None
            try:
                dt = dateparser.parse(f"{date_str} {time_str}")
            except Exception:
                dt = None
            
            rows.append({'datetime': dt, 'author': author, 'mobile': mobile, 'media': media, 'message': message})
        else:
            if rows:
                rows[-1]['message'] += '\n' + line

    df = pd.DataFrame(rows)
    return df
