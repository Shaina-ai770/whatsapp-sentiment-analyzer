
import pandas as pd
import emoji
import re

def extract_emojis(text):
    """
    Extracts all emojis from a given text string and returns them as a list.
    Handles multiple emojis in a single message.
    """
    # The emoji.emoji_list() function is perfect for this.
    # It finds all emoji characters in a string and returns them in a list.
    return [char['emoji'] for char in emoji.emoji_list(text)]

# 1. Sample DataFrame (replace this with your actual chat data)
# This simulates data you would get after parsing a WhatsApp chat file.
chat_data = {
    'user': ['Alice', 'Bob', 'Alice'],
    'message': [
        'Just finished the project! 🎉 So relieved.',
        'Great job! 👍👍 Let’s celebrate 🎏',
        'See you there! 😊'
    ]
}
df = pd.DataFrame(chat_data)

print("--- Original DataFrame ---")
print(df)
print("\n" + "="*30 + "\n")

# 2. The Fix: Apply the emoji extraction function
# We create a new 'emojis' column by applying our function to the 'message' column.
# This is done *before* any cleaning that might remove emojis.
df['emojis'] = df['message'].apply(extract_emojis)

print("--- DataFrame with Extracted Emojis ---")
print(df)
print("\n" + "="*30 + "\n")

# 3. (Optional) Example of a "safe" cleaning function
# This function removes a common URL pattern but leaves emojis intact.
def safe_clean_text(text):
    # Remove URLs - this is safe and doesn't affect emojis
    text = re.sub(r'http\S+', '', text)
    # You can add other safe cleaning steps here
    return text.strip()

# Apply safe cleaning to a different column to preserve the original message
df['message_cleaned'] = df['message'].apply(safe_clean_text)

print("--- DataFrame with Cleaned Text (Emojis Preserved in another column) ---")
print(df[['user', 'message_cleaned', 'emojis']])