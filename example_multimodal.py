"""
Example Usage of Multimodal and Multilingual Sentiment Analysis

This script demonstrates the enhanced features:
1. Multilingual sentiment analysis
2. Language detection
3. Emoji sentiment analysis
4. Multimodal fusion (text + emoji)
5. Emotion detection across languages
"""

from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.multimodal_sentiment import (
    apply_multimodal_sentiment,
    detect_language,
    MultimodalSentimentAnalyzer,
    get_available_models,
    SUPPORTED_LANGUAGES
)
import pandas as pd


def example_language_detection():
    """Example of automatic language detection."""
    print("=" * 70)
    print("Example 1: Language Detection")
    print("=" * 70)

    test_texts = [
        "Hello! How are you today?",
        "Hola! ¿Cómo estás?",
        "Bonjour! Comment allez-vous?",
        "Hallo! Wie geht es dir?",
        "你好! 你好吗?",
        "مرحبا! كيف حالك؟"
    ]

    print("\nDetecting languages:")
    for text in test_texts:
        result = detect_language(text)
        print(f"\nText: {text}")
        print(f"  Detected: {result['language']} (confidence: {result['confidence']:.2%})")


def example_emoji_sentiment():
    """Example of emoji sentiment analysis."""
    print("\n" + "=" * 70)
    print("Example 2: Emoji Sentiment Analysis")
    print("=" * 70)

    analyzer = MultimodalSentimentAnalyzer(language='english')

    test_messages = [
        "I'm so happy! 😊😃🎉",
        "This is terrible 😠😡💔",
        "Feeling okay today 😐",
        "Great news! 🎊✨💪",
        "Very worried about this... 😰😨"
    ]

    print("\nAnalyzing emoji sentiment:")
    for message in test_messages:
        result = analyzer.analyze_multimodal(message)

        print(f"\nMessage: {message}")
        print(f"  Text sentiment: {result['text_sentiment']:.3f}")
        print(f"  Emoji sentiment: {result.get('emoji_sentiment', 0.0):.3f}")
        print(f"  Combined (multimodal): {result['multimodal_sentiment']:.3f}")
        print(f"  Final label: {result['multimodal_label']}")
        print(f"  Emoji count: {result.get('emoji_count', 0)}")


def example_multimodal_fusion():
    """Example showing text-emoji fusion."""
    print("\n" + "=" * 70)
    print("Example 3: Multimodal Fusion (Text + Emoji)")
    print("=" * 70)

    analyzer = MultimodalSentimentAnalyzer(
        language='english',
        enable_emoji_analysis=True,
        enable_multimodal=True
    )

    # Examples where emoji changes the sentiment
    test_cases = [
        ("This is fine 😰", "Text says 'fine' but emoji shows worry"),
        ("Great job! 🙄", "Text is positive but emoji is sarcastic"),
        ("I'm sad 😊", "Text is negative but emoji is positive"),
        ("Terrible news 🎉", "Contradiction between text and emoji")
    ]

    print("\nExamples where emoji modifies text sentiment:")
    for message, description in test_cases:
        result = analyzer.analyze_multimodal(message)

        print(f"\n{description}")
        print(f"  Message: {message}")
        print(f"  Text-only sentiment: {result['text_sentiment']:.3f}")
        print(f"  Emoji sentiment: {result.get('emoji_sentiment', 0.0):.3f}")
        print(f"  Multimodal (70% text + 30% emoji): {result['multimodal_sentiment']:.3f}")
        print(f"  Final label: {result['multimodal_label']}")


def example_multilingual_analysis():
    """Example of multilingual sentiment analysis."""
    print("\n" + "=" * 70)
    print("Example 4: Multilingual Sentiment Analysis")
    print("=" * 70)

    # Show available languages
    print("\nSupported languages:")
    for lang in SUPPORTED_LANGUAGES:
        models = get_available_models(lang)
        print(f"  - {lang}: {models[lang]['sentiment']}")

    # English example
    print("\n1. English Analysis:")
    analyzer_en = MultimodalSentimentAnalyzer(language='english')
    text_en = "This is absolutely fantastic! I love it! 😍"
    result_en = analyzer_en.analyze_multimodal(text_en)
    print(f"   Text: {text_en}")
    print(f"   Sentiment: {result_en['multimodal_label']} ({result_en['multimodal_sentiment']:.3f})")

    # Multilingual model example
    print("\n2. Multilingual Model (works for 100+ languages):")
    analyzer_multi = MultimodalSentimentAnalyzer(language='multilingual')

    texts = {
        'Spanish': "¡Esto es increíble! Me encanta! 🎉",
        'French': "C'est absolument terrible! 😠",
        'German': "Das ist sehr gut! 👍"
    }

    for lang, text in texts.items():
        result = analyzer_multi.analyze_sentiment(text)
        print(f"\n   {lang}: {text}")
        print(f"   Sentiment: {result['label']} (score: {result['score']:.2%})")


def example_batch_analysis():
    """Example with WhatsApp chat data."""
    print("\n" + "=" * 70)
    print("Example 5: Batch Analysis with WhatsApp Chat")
    print("=" * 70)

    # Sample chat with multiple languages and emojis
    sample_chat = """12/31/20, 9:12 PM - Alice: I'm so excited about this project! 😊🎉
12/31/20, 9:13 PM - Bob: ¡Me encanta! Esto es genial! 🌟
12/31/20, 9:14 PM - Charlie: I'm worried about the deadline though 😰
12/31/20, 9:15 PM - Alice: Don't worry! We'll make it! 💪
12/31/20, 9:16 PM - Bob: Oui, c'est possible! 👍
12/31/20, 9:17 PM - Charlie: Thanks everyone! Feeling better now 😊
12/31/20, 9:18 PM - Alice: That's great! 🎊
12/31/20, 9:19 PM - Bob: 加油! 💕"""

    # Parse and preprocess
    print("\n1. Parsing chat...")
    df = parse_chat(sample_chat)
    df = preprocess_df(df)

    # Apply multimodal analysis
    print("2. Applying multimodal analysis...")
    df = apply_multimodal_sentiment(
        df,
        language='multilingual',
        enable_language_detection=True,
        enable_emoji_analysis=True,
        enable_multimodal=True
    )

    # Display results
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)

    for idx, row in df.iterrows():
        print(f"\n{row['author']}: '{row['message']}'")
        print(f"  Detected language: {row.get('detected_language', 'N/A')}")
        print(f"  Sentiment: {row['mm_sentiment']} (score: {row['mm_sentiment_score']:.3f})")
        print(f"  Text sentiment: {row['mm_text_sentiment']:.3f}")
        print(f"  Emoji sentiment: {row.get('mm_emoji_sentiment', 0.0):.3f}")
        print(f"  Emoji count: {row.get('mm_emoji_count', 0)}")
        if 'mm_emotion' in row and row['mm_emotion'] != 'unknown':
            print(f"  Emotion: {row['mm_emotion']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print("=" * 70)

    print(f"\nTotal messages: {len(df)}")

    if 'detected_language' in df.columns:
        print("\nLanguages detected:")
        print(df['detected_language'].value_counts().to_string())

    print("\nSentiment distribution:")
    print(df['mm_sentiment'].value_counts().to_string())

    print(f"\nTotal emojis: {df['mm_emoji_count'].sum()}")
    print(f"Messages with emojis: {(df['mm_emoji_count'] > 0).sum()}")
    print(f"Average emojis per message: {df['mm_emoji_count'].mean():.2f}")

    return df


def example_custom_models():
    """Example using custom models."""
    print("\n" + "=" * 70)
    print("Example 6: Using Custom Models for Specific Languages")
    print("=" * 70)

    print("\nAvailable pre-configured models:")
    all_models = get_available_models('all')

    for lang, models in list(all_models.items())[:5]:  # Show first 5
        print(f"\n{lang.upper()}:")
        print(f"  Sentiment: {models['sentiment']}")
        if models['emotion']:
            print(f"  Emotion: {models['emotion']}")
        else:
            print(f"  Emotion: Not available")

    print("\n" + "=" * 70)
    print("\nYou can also use any HuggingFace model:")
    print("  - Browse models at: https://huggingface.co/models")
    print("  - Filter by task: 'sentiment-analysis' or 'text-classification'")
    print("  - Example custom usage:")
    print("""
    analyzer = MultimodalSentimentAnalyzer(
        sentiment_model="your-custom-model-id",
        emotion_model="your-emotion-model-id"
    )
    """)


def example_preprocessing():
    """Example showing preprocessing for transformers."""
    print("\n" + "=" * 70)
    print("Example 7: Enhanced Preprocessing for Transformers")
    print("=" * 70)

    analyzer = MultimodalSentimentAnalyzer(language='english')

    test_cases = [
        "This is AMAZING!!!!!!!!",
        "I    love    this    sooo    much",
        "😊😊😊😊😊",
        "Check this out: https://example.com/link",
        "Woooooowwwww!!! This is sooooo coooool!!!!"
    ]

    print("\nPreprocessing examples:")
    for text in test_cases:
        preprocessed = analyzer.preprocess_text(text)
        print(f"\nOriginal:     {text}")
        print(f"Preprocessed: {preprocessed}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "Multimodal Sentiment Analysis Examples" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\nNote: First run will download models. This is normal and only happens once.")
    print("Press Ctrl+C to skip any example.\n")

    try:
        input("Press Enter to start examples...")

        # Example 1: Language detection
        example_language_detection()
        input("\nPress Enter for next example...")

        # Example 2: Emoji sentiment
        example_emoji_sentiment()
        input("\nPress Enter for next example...")

        # Example 3: Multimodal fusion
        example_multimodal_fusion()
        input("\nPress Enter for next example...")

        # Example 4: Multilingual
        example_multilingual_analysis()
        input("\nPress Enter for next example...")

        # Example 5: Batch analysis
        df = example_batch_analysis()
        input("\nPress Enter for next example...")

        # Example 6: Custom models
        example_custom_models()
        input("\nPress Enter for next example...")

        # Example 7: Preprocessing
        example_preprocessing()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

        print("\n📝 Try these next:")
        print("  1. Run the Streamlit app: streamlit run streamlit_app.py")
        print("  2. Select 'Multimodal (Text + Emoji)' analysis method")
        print("  3. Upload your own WhatsApp chat")
        print("  4. Explore the 'Multimodal Insights' tab")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")

    except ImportError as e:
        print("\n" + "!" * 70)
        print("ERROR: Required libraries not installed")
        print("!" * 70)
        print(f"\n{e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch langdetect emoji sentencepiece accelerate")
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
