"""
Example Usage of WhatsApp Sentiment Analyzer Core Functions

This script demonstrates how to use all four core functional modules:
1. parse_chat - Parse WhatsApp chat files
2. preprocess_df - Clean and prepare data
3. apply_vader - Perform sentiment analysis
4. save_wordcloud - Generate word cloud visualizations
"""

from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.sentiment import apply_vader, per_user_summary
from src.visualize import save_wordcloud, plot_message_counts, plot_message_counts_by_user
import matplotlib.pyplot as plt


def analyze_whatsapp_chat(chat_file_path):
    """Complete workflow for analyzing WhatsApp chat data.

    Args:
        chat_file_path (str): Path to the exported WhatsApp chat text file

    Returns:
        pd.DataFrame: Fully processed DataFrame with sentiment analysis
    """

    # Step 1: Parse the chat file
    print("Step 1: Parsing WhatsApp chat file...")
    with open(chat_file_path, 'r', encoding='utf-8') as f:
        chat_content = f.read()

    df = parse_chat(chat_content)
    print(f"✓ Parsed {len(df)} messages")
    print(f"  Columns: {list(df.columns)}")

    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing chat data...")
    df = preprocess_df(df)
    print(f"✓ Data cleaned and preprocessed")
    print(f"  New columns: message_raw, urls, emojis, word_count")
    print(f"  Total words: {df['word_count'].sum()}")
    print(f"  Total emojis: {sum(len(e) for e in df['emojis'])}")

    # Step 3: Apply sentiment analysis
    print("\nStep 3: Applying VADER sentiment analysis...")
    df = apply_vader(df)
    print(f"✓ Sentiment analysis complete")
    print(f"  Sentiment distribution:")
    print(df['sentiment'].value_counts().to_string())
    print(f"\n  Average compound score: {df['vader_compound'].mean():.3f}")

    # Step 4: Generate word cloud
    print("\nStep 4: Generating word cloud visualization...")
    all_text = ' '.join(df['message'].dropna())
    save_wordcloud(all_text, 'wordcloud_output.png')
    print(f"✓ Word cloud saved to 'wordcloud_output.png'")

    # Additional: Generate per-user summary
    print("\nBonus: Generating per-user sentiment summary...")
    summary = per_user_summary(df, by=['author'])
    print(summary.to_string())

    return df


def example_with_sample_data():
    """Example using sample chat data (no file required)."""

    print("="*60)
    print("Example: Analyzing Sample WhatsApp Chat Data")
    print("="*60)

    # Sample WhatsApp chat text
    sample_chat = """12/31/20, 9:12 PM - Alice: Hey everyone! Great to be here! 😊
12/31/20, 9:13 PM - Bob: Hi Alice! Welcome to the group
12/31/20, 9:14 PM - Charlie: Hello! Check out this link https://example.com
12/31/20, 9:15 PM - Alice: Thanks! I'm so excited about this project 🎉
12/31/20, 9:16 PM - Bob: Yeah, it's going to be amazing!
12/31/20, 9:17 PM - Charlie: I'm a bit worried about the deadline though 😰
12/31/20, 9:18 PM - Alice: Don't worry, we'll make it work together!
12/31/20, 9:19 PM - Bob: Absolutely! Team effort!
12/31/20, 9:20 PM - Charlie: You guys are the best! 💪"""

    # Step 1: Parse
    print("\n1. Parsing chat...")
    df = parse_chat(sample_chat)
    print(f"   Parsed {len(df)} messages from {df['author'].nunique()} users")

    # Step 2: Preprocess
    print("\n2. Preprocessing...")
    df = preprocess_df(df)
    print(f"   Extracted {sum(len(e) for e in df['emojis'])} emojis")
    print(f"   Found {sum(len(u) for u in df['urls'])} URLs")

    # Step 3: Sentiment Analysis
    print("\n3. Analyzing sentiment...")
    df = apply_vader(df)

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\nMessage-level sentiment:")
    for idx, row in df.iterrows():
        print(f"  {row['author']}: '{row['message'][:40]}...' → {row['sentiment'].upper()} ({row['vader_compound']:.2f})")

    print("\nOverall sentiment distribution:")
    print(df['sentiment'].value_counts().to_string())

    print("\nPer-user sentiment summary:")
    summary = per_user_summary(df, by=['author'])
    print(summary.to_string())

    # Step 4: Word cloud
    print("\n4. Generating word cloud...")
    all_text = ' '.join(df['message'].dropna())
    save_wordcloud(all_text, 'sample_wordcloud.png')
    print(f"   ✓ Word cloud saved to 'sample_wordcloud.png'")

    return df


if __name__ == "__main__":
    # Run example with sample data (no file needed)
    df = example_with_sample_data()

    # Uncomment below to analyze your own WhatsApp chat export:
    # df = analyze_whatsapp_chat('path/to/your/chat_export.txt')

    print("\n" + "="*60)
    print("Analysis complete! Check the output files and DataFrame.")
    print("="*60)
