# WhatsApp Group Sentiment Analyzer

A comprehensive Python project to parse exported WhatsApp group chats, perform sentiment analysis (VADER & Transformers), detect emotions, and generate visualizations. Includes a Streamlit app for interactive analysis.

## Features

### Core Capabilities
- **Chat Parsing**: Supports multiple WhatsApp export formats
- **Interactive Dashboard**: Streamlit-based web interface with multiple tabs
- **Visualizations**: Word clouds, sentiment trends, per-user analysis
- **Export & Analysis**: Generate reports and export processed data

### Sentiment Analysis Methods
- **VADER** (fast, traditional lexicon-based)
- **Transformers** (advanced, BERT/RoBERTa-based)
- **Multimodal** (text + emoji analysis) ⭐ NEW

### Advanced Features
- **Multilingual Support**: 100+ languages with automatic detection
- **Emotion Detection**: Joy, sadness, anger, fear, surprise, love, and more
- **Emoji Analysis**: Sentiment analysis from emojis with 60+ emoji mappings
- **Multimodal Fusion**: Intelligent combination of text and emoji sentiment
- **Language Detection**: Automatic language identification per message

## Quick Start

### 1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies:

For basic features (VADER only):
```bash
pip install pandas numpy nltk vaderSentiment matplotlib wordcloud streamlit
```

For advanced features (includes Transformers):
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### 4. Upload and analyze:

- Export your WhatsApp chat (Chat > More > Export Chat)
- Upload the .txt file in the Streamlit app
- Choose analysis method (VADER or Transformers)
- Explore the results!

## Usage Examples

### Python API

```python
from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.sentiment import apply_vader
from src.advanced_sentiment import apply_advanced_sentiment
from src.multimodal_sentiment import apply_multimodal_sentiment

# Parse WhatsApp chat
with open('chat.txt', 'r') as f:
    df = parse_chat(f.read())

# Preprocess
df = preprocess_df(df)

# Option 1: Fast VADER sentiment
df = apply_vader(df)

# Option 2: Advanced transformer sentiment + emotions
df = apply_advanced_sentiment(df, include_emotions=True)

# Option 3: Multimodal (text + emoji) with multilingual support ⭐ NEW
df = apply_multimodal_sentiment(
    df,
    language='multilingual',
    enable_language_detection=True,
    enable_emoji_analysis=True,
    enable_multimodal=True
)

# View results
print(df[['message', 'mm_sentiment', 'mm_emotion', 'detected_language']].head())
```

### Run Examples

```bash
# Basic usage example
python example_usage.py

# Advanced sentiment analysis example
python example_advanced_sentiment.py

# Multimodal & multilingual example ⭐ NEW
python example_multimodal.py
```

## Project Layout

### Core Modules
- `src/parser.py`: Parse exported WhatsApp chat into DataFrame
- `src/preprocess.py`: Clean messages, extract emojis, URLs, and word counts
- `src/sentiment.py`: Apply VADER sentiment analysis
- `src/advanced_sentiment.py`: Transformer-based sentiment and emotion analysis
- `src/multimodal_sentiment.py`: ⭐ **NEW** - Multimodal & multilingual analysis
- `src/visualize.py`: Generate charts and word clouds

### Applications
- `streamlit_app.py`: Interactive web dashboard (updated with multimodal features)
- `example_usage.py`: Basic usage examples
- `example_advanced_sentiment.py`: Advanced analysis examples
- `example_multimodal.py`: ⭐ **NEW** - Multimodal & multilingual examples

### Tests
- `tests/`: Unit tests for core functionality

### Documentation
- `README.md`: This file
- `ADVANCED_SENTIMENT_GUIDE.md`: Comprehensive guide for transformer features
- `MULTIMODAL_GUIDE.md`: ⭐ **NEW** - Complete guide for multimodal features

## Advanced Features

### Sentiment Analysis Methods

| Method | Speed | Accuracy | Emotions | Multilingual | Emoji Support | Best For |
|--------|-------|----------|----------|--------------|---------------|----------|
| **VADER** | Very Fast | Good | No | No | Basic | Quick analysis |
| **Transformers** | Moderate | Excellent | Yes | Limited | No | Accurate text analysis |
| **Multimodal** ⭐ | Moderate | Excellent+ | Yes | 100+ languages | Advanced | Comprehensive context |

### Emotion Detection

When using transformer or multimodal models, detect specific emotions:
- Joy / Happiness
- Sadness
- Anger
- Fear / Anxiety
- Surprise
- Love
- Disgust
- And more...

### Multilingual Support ⭐

Analyze chats in **100+ languages**:
- **Pre-configured models** for: English, Spanish, French, German, Arabic, Chinese, Hindi
- **Automatic language detection** for mixed-language chats
- **Multilingual model** works across all languages

### Emoji Analysis ⭐

Advanced emoji sentiment analysis:
- **60+ emoji mappings** with calibrated sentiment scores
- **Multimodal fusion**: Combine text (70%) and emoji (30%) sentiment
- **Disagreement detection**: Identify sarcasm and conflicting signals
- **Per-user emoji patterns**: Track emoji usage by participant

### Streamlit Dashboard Features

- **Three Analysis Methods**: VADER, Transformers, or Multimodal
- **Filter by**: Date range, user, sentiment, emotion, and language
- **Interactive visualizations**:
  - Sentiment distribution over time
  - Per-user sentiment and emotion analysis
  - Emoji usage patterns ⭐
  - Word clouds
  - Language distribution ⭐
- **Advanced Analytics Tab**:
  - Model comparison and confidence scores
  - Detailed emotion breakdowns
  - High-confidence predictions
- **Multimodal Insights Tab** ⭐:
  - Language detection results
  - Emoji sentiment analysis
  - Text vs emoji comparison
  - Sentiment disagreement detection
- **Export analyzed data** with all metrics

## Documentation

- **[Multimodal Guide](MULTIMODAL_GUIDE.md)** ⭐: Complete guide for multimodal & multilingual features
- **[Advanced Sentiment Guide](ADVANCED_SENTIMENT_GUIDE.md)**: Complete guide for transformer-based analysis
- **API Reference**: See docstrings in each module
- **Examples**: Check `example_usage.py`, `example_advanced_sentiment.py`, and `example_multimodal.py`

## Performance Tips

- Use VADER for datasets > 10,000 messages
- Use Transformers with GPU for best performance
- First transformer run downloads models (~500MB)
- Subsequent runs are cached and much faster

## Requirements

### Basic Installation
- Python 3.8+
- pandas, numpy
- nltk, vaderSentiment
- matplotlib, wordcloud
- streamlit

### Advanced Features (Optional)
- transformers >= 4.30.0
- torch >= 2.0.0
- sentencepiece
- accelerate
- langdetect (for language detection) ⭐
- emoji >= 2.0.0 (for emoji analysis) ⭐

## Troubleshooting

### Transformers/Multimodal Not Working?
```bash
pip install transformers torch sentencepiece accelerate langdetect emoji
```

### Models Not Downloading?
- Check internet connection
- Clear cache: `rm -rf ~/.cache/huggingface/`
- Manually download: `python -c "from transformers import pipeline; pipeline('sentiment-analysis')"`

### Language Detection Errors?
```bash
pip install --upgrade langdetect
```

### Emoji Not Detected?
```bash
pip install --upgrade emoji>=2.0.0
```

### Out of Memory?
- Use VADER instead
- Use language-specific model instead of multilingual
- Process smaller batches
- Disable language detection
- Use CPU instead of GPU

See [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md) and [ADVANCED_SENTIMENT_GUIDE.md](ADVANCED_SENTIMENT_GUIDE.md) for more troubleshooting.

## Contributing

Contributions welcome! Areas for improvement:
- Additional language support
- More pre-trained models
- Performance optimizations
- Additional visualizations
- Bug fixes and documentation

## License

MIT License - See LICENSE file for details

## Citation

If you use this project in research, please cite the underlying models:
- VADER: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- RoBERTa Sentiment: Barbieri et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.
- Emotion Detection: Hartmann et al. (2022). More than a Feeling: Accuracy and Application of Sentiment Analysis.

## Support

- Read the documentation
- Check example scripts
- Review troubleshooting section
- Open an issue on GitHub

---

**Note**: First run with transformer models will download ~500MB of model files. This is normal and only happens once.
