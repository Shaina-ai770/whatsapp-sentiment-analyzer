"""
Verify that all multimodal dependencies are installed correctly.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name}: Installed")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT installed")
        return False

def main():
    print("=" * 60)
    print("Verifying Multimodal Dependencies")
    print("=" * 60)

    required_packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("langdetect", "langdetect"),
        ("emoji", "emoji"),
        ("sentencepiece", "sentencepiece"),
        ("accelerate", "accelerate")
    ]

    all_installed = True

    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False

    print("\n" + "=" * 60)

    if all_installed:
        print("✅ All packages installed successfully!")
        print("\nYou can now use multimodal features:")
        print("  1. Run: streamlit run streamlit_app.py")
        print("  2. Select 'Multimodal (Text + Emoji)' in sidebar")
        print("  3. Upload your WhatsApp chat file")

        # Test if multimodal module works
        print("\n" + "=" * 60)
        print("Testing multimodal module...")
        try:
            from src.multimodal_sentiment import MultimodalSentimentAnalyzer
            print("✅ Multimodal module loads successfully!")

            print("\nSupported languages:")
            from src.multimodal_sentiment import SUPPORTED_LANGUAGES
            for lang in SUPPORTED_LANGUAGES:
                print(f"  - {lang}")

        except Exception as e:
            print(f"⚠️ Warning: {e}")

    else:
        print("❌ Some packages are missing!")
        print("\nInstall missing packages with:")
        print("  pip install transformers torch sentencepiece accelerate langdetect emoji")

    print("=" * 60)

    return 0 if all_installed else 1

if __name__ == "__main__":
    sys.exit(main())
