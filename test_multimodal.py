"""
Quick test script for multimodal features.
Run this AFTER installing all dependencies.
"""

def test_multimodal():
    """Test that multimodal features work."""
    print("=" * 60)
    print("Testing Multimodal Sentiment Analysis")
    print("=" * 60)

    try:
        from src.multimodal_sentiment import MultimodalSentimentAnalyzer, detect_language

        print("\n✅ Module imported successfully!")

        # Test language detection
        print("\n1. Testing language detection...")
        result = detect_language("Hello, how are you?")
        print(f"   Language detected: {result['language']} (confidence: {result['confidence']:.2%})")

        # Test analyzer initialization
        print("\n2. Initializing analyzer...")
        print("   NOTE: First run will download models (~500MB). This may take 5-10 minutes.")
        print("   Subsequent runs will be instant (models are cached).")

        analyzer = MultimodalSentimentAnalyzer(
            language='english',
            enable_emoji_analysis=True,
            enable_multimodal=True
        )

        print("   ✅ Analyzer initialized!")

        # Test sentiment analysis
        print("\n3. Testing sentiment analysis...")
        test_message = "I'm so happy! 😊🎉"
        result = analyzer.analyze_multimodal(test_message)

        print(f"   Message: {test_message}")
        print(f"   Sentiment: {result['multimodal_label']}")
        print(f"   Score: {result['multimodal_sentiment']:.2f}")
        print(f"   Emoji sentiment: {result.get('emoji_sentiment', 0):.2f}")
        print(f"   Emoji count: {result.get('emoji_count', 0)}")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("\nMultimodal features are working correctly!")
        print("\nNext steps:")
        print("  1. Run: streamlit run streamlit_app.py")
        print("  2. Select 'Multimodal (Text + Emoji)' in sidebar")
        print("  3. Choose your language or 'multilingual'")
        print("  4. Enable desired options")
        print("  5. Upload your WhatsApp chat file")
        print("=" * 60)

        return True

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPlease install missing packages:")
        print("  pip install transformers torch sentencepiece accelerate langdetect")
        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be a temporary issue. Try:")
        print("  1. Check your internet connection")
        print("  2. Clear cache: rm -rf ~/.cache/huggingface/")
        print("  3. Run again: python test_multimodal.py")
        return False

if __name__ == "__main__":
    success = test_multimodal()
    exit(0 if success else 1)
