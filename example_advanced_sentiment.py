"""
Example Usage of Advanced Sentiment Analysis with Transformers

This script demonstrates how to use the advanced sentiment analysis module
with transformer-based models (BERT/RoBERTa) for more accurate sentiment
detection and emotion recognition.
"""

from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.sentiment import apply_vader
from src.advanced_sentiment import (
    apply_advanced_sentiment,
    compare_sentiments,
    quick_analyze,
    AdvancedSentimentAnalyzer
)
import pandas as pd


def example_quick_analysis():
    """Quick analysis of individual messages."""
    print("=" * 70)
    print("Example 1: Quick Analysis of Individual Messages")
    print("=" * 70)

    test_messages = [
        "I'm so excited about this project! It's going to be amazing! 🎉",
        "I'm really worried about the deadline... 😰",
        "The meeting was okay, nothing special.",
        "I can't believe this happened! This is terrible! 😠",
        "Thank you so much! You're the best! ❤️"
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\nMessage {i}: {message}")
        result = quick_analyze(message, include_emotion=True)

        sentiment = result['sentiment']
        emotion = result['emotion']

        print(f"  Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2%})")
        print(f"  Emotion: {emotion['emotion']} (confidence: {emotion['score']:.2%})")

        # Show top 3 emotions
        if emotion['all_emotions']:
            top_emotions = sorted(
                emotion['all_emotions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print(f"  Top emotions: {', '.join([f'{e}:{s:.2f}' for e, s in top_emotions])}")


def example_batch_analysis():
    """Batch analysis with sample WhatsApp data."""
    print("\n" + "=" * 70)
    print("Example 2: Batch Analysis with WhatsApp Chat")
    print("=" * 70)

    # Sample WhatsApp chat
    sample_chat = """12/31/20, 9:12 PM - Alice: I'm so happy we're finally doing this! 😊
12/31/20, 9:13 PM - Bob: Yeah, but I'm nervous about the presentation tomorrow
12/31/20, 9:14 PM - Charlie: Don't worry! We've practiced enough
12/31/20, 9:15 PM - Alice: Absolutely! We're going to crush it! 💪
12/31/20, 9:16 PM - Bob: I guess you're right... still feeling anxious though
12/31/20, 9:17 PM - Charlie: That's normal! Channel that energy into excitement!
12/31/20, 9:18 PM - Alice: Exactly! We got this team! 🎉
12/31/20, 9:19 PM - Bob: Thanks guys, you're amazing!
12/31/20, 9:20 PM - Charlie: Love you all! See you tomorrow! ❤️"""

    # Parse and preprocess
    print("\n1. Parsing chat...")
    df = parse_chat(sample_chat)
    df = preprocess_df(df)

    # Apply both VADER and advanced sentiment
    print("2. Applying VADER sentiment analysis...")
    df = apply_vader(df)

    print("3. Applying advanced transformer sentiment analysis...")
    print("   (This may take a minute on first run as models download)")
    df = apply_advanced_sentiment(df, include_emotions=True)

    # Display results
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)

    for idx, row in df.iterrows():
        print(f"\n{row['author']}: '{row['message'][:50]}...'")
        print(f"  VADER: {row['sentiment']} (compound: {row['vader_compound']:.2f})")
        print(f"  Transformer: {row['transformer_sentiment']} "
              f"(confidence: {row['transformer_sentiment_score']:.2%})")
        print(f"  Emotion: {row['transformer_emotion']} "
              f"(confidence: {row['transformer_emotion_score']:.2%})")

    # Show comparison
    print("\n" + "=" * 70)
    print("Model Comparison:")
    print("=" * 70)
    comparison = compare_sentiments(df)
    print(comparison.to_string(index=False))

    # Show emotion distribution
    print("\n" + "=" * 70)
    print("Emotion Distribution:")
    print("=" * 70)
    emotion_dist = df['transformer_emotion'].value_counts()
    print(emotion_dist.to_string())

    return df


def example_per_user_analysis(df):
    """Per-user sentiment and emotion analysis."""
    print("\n" + "=" * 70)
    print("Example 3: Per-User Analysis")
    print("=" * 70)

    # Sentiment by user
    print("\nSentiment Distribution by User:")
    sentiment_by_user = df.groupby('author')['transformer_sentiment'].value_counts().unstack(fill_value=0)
    print(sentiment_by_user.to_string())

    # Emotion by user
    print("\nEmotion Distribution by User:")
    emotion_by_user = df.groupby('author')['transformer_emotion'].value_counts().unstack(fill_value=0)
    print(emotion_by_user.to_string())

    # Average confidence by user
    print("\nAverage Confidence Scores by User:")
    confidence_by_user = df.groupby('author').agg({
        'transformer_sentiment_score': 'mean',
        'transformer_emotion_score': 'mean'
    })
    print(confidence_by_user.to_string())


def example_custom_models():
    """Example using custom models."""
    print("\n" + "=" * 70)
    print("Example 4: Using Custom Models")
    print("=" * 70)

    print("\nAvailable pre-trained models you can try:")
    print("  Sentiment Analysis:")
    print("    - cardiffnlp/twitter-roberta-base-sentiment-latest (default)")
    print("    - distilbert-base-uncased-finetuned-sst-2-english")
    print("    - nlptown/bert-base-multilingual-uncased-sentiment")
    print("\n  Emotion Detection:")
    print("    - j-hartmann/emotion-english-distilroberta-base (default)")
    print("    - bhadresh-savani/distilbert-base-uncased-emotion")
    print("    - SamLowe/roberta-base-go_emotions")

    # Example with custom model
    print("\nInitializing with custom sentiment model...")
    try:
        analyzer = AdvancedSentimentAnalyzer(
            sentiment_model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        test_text = "This is absolutely fantastic!"
        result = analyzer.analyze_sentiment(test_text)

        print(f"Text: {test_text}")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.2%})")
    except Exception as e:
        print(f"Note: {e}")
        print("Custom model example requires internet connection for first download")


def example_error_handling():
    """Example showing robust error handling."""
    print("\n" + "=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)

    analyzer = AdvancedSentimentAnalyzer()

    # Test with various edge cases
    test_cases = [
        ("", "Empty string"),
        (None, "None value"),
        ("   ", "Whitespace only"),
        ("a" * 1000, "Very long text (will be truncated)"),
        ("😊😊😊", "Emojis only"),
    ]

    for text, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {repr(text)[:50]}")
        try:
            result = analyzer.analyze_sentiment(str(text) if text is not None else "")
            print(f"  Sentiment: {result['label']} (confidence: {result['score']:.2%})")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "Advanced Sentiment Analysis Examples" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\nNote: First run will download models (~500MB), which may take a few minutes.")
    print("Subsequent runs will be much faster as models are cached.\n")

    input("Press Enter to start examples...")

    try:
        # Example 1: Quick analysis
        example_quick_analysis()

        input("\nPress Enter to continue to next example...")

        # Example 2: Batch analysis
        df = example_batch_analysis()

        input("\nPress Enter to continue to next example...")

        # Example 3: Per-user analysis
        example_per_user_analysis(df)

        input("\nPress Enter to continue to next example...")

        # Example 4: Custom models
        example_custom_models()

        input("\nPress Enter to continue to next example...")

        # Example 5: Error handling
        example_error_handling()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except ImportError as e:
        print("\n" + "!" * 70)
        print("ERROR: Required libraries not installed")
        print("!" * 70)
        print(f"\n{e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch sentencepiece accelerate")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")

    except Exception as e:
        print("\n" + "!" * 70)
        print("ERROR: An unexpected error occurred")
        print("!" * 70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
