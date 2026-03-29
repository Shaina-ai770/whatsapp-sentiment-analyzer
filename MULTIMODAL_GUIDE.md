## Multimodal and Multilingual Sentiment Analysis Guide

## Overview

This guide covers the enhanced multimodal and multilingual sentiment analysis features of the WhatsApp Sentiment Analyzer. These advanced capabilities go beyond basic text analysis to provide comprehensive sentiment understanding across languages, modalities, and contexts.

## Key Features

### 1. **Multilingual Support**
- Supports **100+ languages** using multilingual transformer models
- Automatic language detection for each message
- Language-specific models for optimal accuracy
- Pre-configured models for 8+ major languages

### 2. **Multimodal Analysis**
- Analyzes both **text and emojis** together
- Emoji sentiment scoring with 60+ emoji mappings
- Intelligent fusion of text and emoji sentiment (70% text + 30% emoji)
- Identifies sentiment disagreements between text and emojis

### 3. **Enhanced Preprocessing**
- Transformer-optimized text cleaning
- Emoji preservation for multimodal analysis
- Normalization of repeated characters and punctuation
- URL handling and whitespace normalization

### 4. **Emotion Detection**
- Detects specific emotions (joy, sadness, anger, fear, surprise, etc.)
- Multi-label emotion scoring
- Language-specific emotion models where available

### 5. **Robust Architecture**
- Modular design with clear separation of concerns
- Comprehensive error handling
- GPU acceleration support
- Batch processing for efficiency

## Installation

### Required Dependencies

```bash
pip install transformers>=4.30.0 torch>=2.0.0 langdetect emoji>=2.0.0 sentencepiece accelerate
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced visualizations in Streamlit:
```bash
pip install plotly
```

## Quick Start

### Python API

```python
from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.multimodal_sentiment import apply_multimodal_sentiment

# Parse chat
with open('chat.txt', 'r') as f:
    df = parse_chat(f.read())

# Preprocess
df = preprocess_df(df)

# Apply multimodal analysis
df = apply_multimodal_sentiment(
    df,
    language='multilingual',
    enable_language_detection=True,
    enable_emoji_analysis=True,
    enable_multimodal=True
)

# Access results
print(df[['message', 'mm_sentiment', 'mm_emotion', 'detected_language']].head())
```

### Streamlit App

```bash
streamlit run streamlit_app.py
```

Then:
1. Select **"Multimodal (Text + Emoji)"** in the sidebar
2. Choose your language or use "multilingual"
3. Enable desired options (language detection, emoji analysis, etc.)
4. Upload your WhatsApp chat file
5. Explore the **"Multimodal Insights"** tab

## Supported Languages

### Pre-configured Language Models

| Language | Sentiment Model | Emotion Model |
|----------|-----------------|---------------|
| **English** | twitter-roberta-base-sentiment-latest | emotion-english-distilroberta-base |
| **Multilingual** | bert-base-multilingual-uncased-sentiment | xlm-roberta-large-xnli |
| **Spanish** | beto-sentiment-analysis | N/A |
| **French** | bert-base-multilingual-uncased-sentiment | N/A |
| **German** | german-sentiment-bert | N/A |
| **Arabic** | bert-base-arabic-camelbert-mix-sentiment | N/A |
| **Chinese** | roberta-base-finetuned-jd-binary-chinese | N/A |
| **Hindi** | bert-base-multilingual-uncased-sentiment | N/A |

### Using Multilingual Model

The **multilingual** option uses a BERT model fine-tuned on 100+ languages. While slightly less accurate than language-specific models, it works across languages and is ideal for:
- Mixed-language chats
- Unknown languages
- Languages without specific models

## Features in Detail

### 1. Language Detection

Automatically detect the language of each message:

```python
from src.multimodal_sentiment import detect_language

text = "Hello, how are you?"
result = detect_language(text)

print(result['language'])      # 'en'
print(result['confidence'])    # 0.99
print(result['all_languages']) # [{'lang': 'en', 'prob': 0.99}, ...]
```

In DataFrame analysis:

```python
df = apply_multimodal_sentiment(
    df,
    enable_language_detection=True
)

# Access language data
print(df[['message', 'detected_language', 'language_confidence']])
```

### 2. Emoji Sentiment Analysis

Analyze sentiment from emojis with pre-defined sentiment mappings:

```python
analyzer = MultimodalSentimentAnalyzer(
    language='english',
    enable_emoji_analysis=True
)

result = analyzer.analyze_multimodal("I'm happy! 😊🎉")

print(result['emoji_sentiment'])      # 0.825 (average of emoji scores)
print(result['emoji_count'])          # 2
print(result['positive_emojis'])      # 2
print(result['negative_emojis'])      # 0
```

**Emoji Sentiment Dictionary (Sample)**:
- 😊 → 0.8 (positive)
- 😍 → 0.95 (very positive)
- 😢 → -0.8 (negative)
- 😠 → -0.8 (negative)
- 🎉 → 0.85 (positive)
- 😐 → 0.0 (neutral)

Full dictionary includes 60+ emojis with calibrated sentiment scores.

### 3. Multimodal Fusion

Combine text and emoji sentiment for more accurate analysis:

```python
analyzer = MultimodalSentimentAnalyzer(
    language='english',
    enable_multimodal=True
)

# Example where emoji changes the sentiment
result = analyzer.analyze_multimodal("This is fine 😰")

print(result['text_sentiment'])        # 0.3 (slightly positive text)
print(result['emoji_sentiment'])       # -0.7 (worried emoji)
print(result['multimodal_sentiment'])  # 0.0 (70% text + 30% emoji)
print(result['multimodal_label'])      # 'neutral' (combined assessment)
```

**Fusion Formula**:
```
multimodal_sentiment = 0.7 × text_sentiment + 0.3 × emoji_sentiment
```

This weighting reflects that text usually carries more semantic content, while emojis provide emotional context.

### 4. Enhanced Preprocessing

Text is preprocessed specifically for transformer models:

```python
analyzer = MultimodalSentimentAnalyzer()

# Original text
text = "This is AMAZING!!!!!!! Woooow 😊"

# Preprocessed text
preprocessed = analyzer.preprocess_text(text)
# "This is AMAZING!! Woow 😊"

# Features:
# - Excessive punctuation reduced (!!!!! → !!)
# - Repeated characters normalized (oooo → oo)
# - Whitespace normalized
# - Emojis preserved (for multimodal analysis)
# - URLs removed
```

### 5. Custom Models

Use any HuggingFace model:

```python
from src.multimodal_sentiment import MultimodalSentimentAnalyzer

analyzer = MultimodalSentimentAnalyzer(
    sentiment_model="nlptown/bert-base-multilingual-uncased-sentiment",
    emotion_model="your-custom-emotion-model",
    device='cuda'  # Use GPU if available
)
```

Browse models at: https://huggingface.co/models?pipeline_tag=sentiment-analysis

## API Reference

### MultimodalSentimentAnalyzer

Main class for multimodal analysis.

```python
class MultimodalSentimentAnalyzer:
    def __init__(
        self,
        language: str = 'english',
        sentiment_model: Optional[str] = None,
        emotion_model: Optional[str] = None,
        device: Optional[str] = None,
        enable_emoji_analysis: bool = True,
        enable_multimodal: bool = True
    )
```

**Methods:**
- `preprocess_text(text)` → Enhanced text preprocessing
- `extract_emojis(text)` → Extract emojis from text
- `analyze_emoji_sentiment(emojis)` → Analyze emoji sentiment
- `analyze_sentiment(text)` → Analyze text sentiment
- `analyze_multimodal(text)` → Combined text + emoji analysis
- `analyze_emotion(text)` → Detect emotions

### apply_multimodal_sentiment

Apply multimodal analysis to DataFrame.

```python
def apply_multimodal_sentiment(
    df: pd.DataFrame,
    text_column: str = 'message',
    language: str = 'english',
    sentiment_model: Optional[str] = None,
    emotion_model: Optional[str] = None,
    enable_language_detection: bool = False,
    enable_emoji_analysis: bool = True,
    enable_multimodal: bool = True
) -> pd.DataFrame
```

**Returns DataFrame with columns:**
- `mm_sentiment` - Sentiment label (positive/negative/neutral)
- `mm_sentiment_score` - Multimodal sentiment score (-1 to 1)
- `mm_text_sentiment` - Text-only sentiment score
- `mm_confidence` - Model confidence (0-1)
- `mm_emoji_sentiment` - Emoji-only sentiment score
- `mm_emoji_count` - Number of emojis
- `mm_positive_emojis` - Count of positive emojis
- `mm_negative_emojis` - Count of negative emojis
- `mm_emotion` - Primary emotion detected
- `mm_emotion_score` - Emotion confidence
- `mm_all_emotions` - Dictionary of all emotion scores
- `detected_language` - Detected language code (if enabled)
- `language_confidence` - Language detection confidence (if enabled)

### detect_language

Detect language of text.

```python
def detect_language(text: str) -> Dict[str, any]
```

Returns dictionary with:
- `language` - ISO language code (e.g., 'en', 'es', 'fr')
- `confidence` - Detection confidence (0-1)
- `all_languages` - List of detected languages with probabilities

### get_available_models

Get available pre-configured models.

```python
def get_available_models(language: str = 'all') -> Dict[str, Dict]
```

## Usage Examples

### Example 1: Basic Multimodal Analysis

```python
from src.multimodal_sentiment import MultimodalSentimentAnalyzer

analyzer = MultimodalSentimentAnalyzer(language='english')

messages = [
    "I'm so happy! 😊",
    "This is terrible 😠",
    "Just okay 😐"
]

for msg in messages:
    result = analyzer.analyze_multimodal(msg)
    print(f"{msg} → {result['multimodal_label']} ({result['multimodal_sentiment']:.2f})")
```

### Example 2: Multilingual Chat Analysis

```python
# Chat with multiple languages
df = apply_multimodal_sentiment(
    df,
    language='multilingual',
    enable_language_detection=True
)

# View language distribution
print(df['detected_language'].value_counts())

# Filter by language
spanish_messages = df[df['detected_language'] == 'es']
print(spanish_messages[['message', 'mm_sentiment']])
```

### Example 3: Emoji-Heavy Conversations

```python
# Analyze emoji usage patterns
df = apply_multimodal_sentiment(
    df,
    enable_emoji_analysis=True,
    enable_multimodal=True
)

# Find messages with many emojis
emoji_rich = df[df['mm_emoji_count'] >= 3]

# Find sentiment disagreements (text vs emoji)
disagreements = df[
    (df['mm_text_sentiment'] * df['mm_emoji_sentiment'] < 0) &
    (abs(df['mm_text_sentiment']) > 0.3)
]

print("Messages where text and emoji sentiments disagree:")
print(disagreements[['message', 'mm_text_sentiment', 'mm_emoji_sentiment']])
```

### Example 4: Language-Specific Analysis

```python
# Use Spanish-specific model
analyzer = MultimodalSentimentAnalyzer(language='spanish')

spanish_text = "¡Esto es increíble! Me encanta mucho! 🎉"
result = analyzer.analyze_multimodal(spanish_text)

print(f"Sentiment: {result['multimodal_label']}")
print(f"Score: {result['multimodal_sentiment']:.2f}")
```

### Example 5: Per-User Emoji Analysis

```python
# Analyze emoji usage by user
emoji_stats = df.groupby('author').agg({
    'mm_emoji_count': 'sum',
    'mm_positive_emojis': 'sum',
    'mm_negative_emojis': 'sum'
})

# Calculate emoji positivity ratio
emoji_stats['positivity_ratio'] = (
    emoji_stats['mm_positive_emojis'] /
    (emoji_stats['mm_positive_emojis'] + emoji_stats['mm_negative_emojis'])
)

print(emoji_stats.sort_values('positivity_ratio', ascending=False))
```

## Streamlit Dashboard Features

When using the Streamlit app with multimodal analysis:

### Sidebar Options

1. **Analysis Method Selection**
   - Choose "Multimodal (Text + Emoji)"

2. **Language Selection**
   - Select target language or "multilingual"
   - Auto-detect language per message

3. **Multimodal Options**
   - Enable/disable emoji sentiment analysis
   - Toggle multimodal fusion
   - Enable/disable emotion detection

### Multimodal Insights Tab

Exclusive tab showing:

1. **Language Detection Results**
   - Language distribution chart
   - Confidence metrics
   - Sample messages by language

2. **Emoji Analysis**
   - Total emoji count
   - Average emojis per message
   - Emoji sentiment distribution
   - Per-user emoji usage statistics

3. **Text vs Emoji Comparison**
   - Correlation between text and emoji sentiment
   - Messages with sentiment disagreements
   - Examples where emojis change meaning

4. **Model Performance**
   - Average confidence scores
   - Overall sentiment summary
   - Most common emotions

## Performance Considerations

### Speed Benchmarks

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| VADER | ~1000 msg/s | Good | Quick exploration |
| Transformers | ~10-50 msg/s | Excellent | Accurate analysis |
| Multimodal | ~10-50 msg/s | Excellent | Full context |

### Memory Requirements

- **Multilingual model**: ~500MB
- **Language-specific model**: ~300-500MB
- **Emoji processing**: Minimal
- **Language detection**: ~5MB

### Optimization Tips

1. **Use language-specific models** when possible (faster than multilingual)
2. **Disable language detection** if language is known
3. **Process in batches** for large datasets
4. **Use GPU** for 10-50x speedup
5. **Cache results** to avoid reprocessing

## Comparison: Methods Overview

| Feature | VADER | Transformers | Multimodal |
|---------|-------|--------------|------------|
| **Speed** | Very Fast | Moderate | Moderate |
| **Accuracy** | Good | Excellent | Excellent+ |
| **Languages** | English | 1-100+ | 1-100+ |
| **Emoji Support** | Basic | No | Advanced |
| **Context Aware** | No | Yes | Yes |
| **Emotion Detection** | No | Yes | Yes |
| **Multimodal** | No | No | Yes |
| **Setup** | Simple | Model download | Model download |
| **Best For** | Quick analysis | Accurate text | Full context |

## Troubleshooting

### Models Not Downloading

```bash
# Check internet connection
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Manually download
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

### Language Detection Errors

```python
# If langdetect fails
pip install --upgrade langdetect

# For very short texts, language detection may be unreliable
# Use known language instead
df = apply_multimodal_sentiment(df, language='english', enable_language_detection=False)
```

### Out of Memory

```python
# Use smaller batch sizes
# Process subset of data
df_sample = df.head(100)

# Use language-specific model (smaller than multilingual)
analyzer = MultimodalSentimentAnalyzer(language='english')

# Disable emoji analysis if not needed
df = apply_multimodal_sentiment(df, enable_emoji_analysis=False)
```

### Emoji Not Detected

```bash
# Install/upgrade emoji library
pip install --upgrade emoji>=2.0.0

# Check emoji in your data
df[df['emojis'].str.len() > 0]
```

### Slow Performance

```python
# Use GPU if available
analyzer = MultimodalSentimentAnalyzer(device='cuda')

# Disable language detection
enable_language_detection=False

# Use English model (fastest)
language='english'
```

## Best Practices

### 1. **Language Selection**
- Use **language-specific** models when language is known
- Use **multilingual** for mixed-language chats
- Enable **language detection** only when needed

### 2. **Emoji Analysis**
- Always enable for social media/informal chats
- Disable for formal communications
- Check for text-emoji disagreements for sarcasm detection

### 3. **Multimodal Fusion**
- Enable for comprehensive sentiment
- Disable if emojis are decorative only
- Adjust weights if needed (modify source code)

### 4. **Preprocessing**
- Trust the built-in preprocessing
- Don't remove emojis before analysis
- Keep URLs for context (they're handled internally)

### 5. **Model Selection**
- Start with pre-configured models
- Use custom models for specialized domains
- Test multiple models for best results

## Advanced Topics

### Custom Emoji Sentiment Dictionary

Modify `EMOJI_SENTIMENT` in `src/multimodal_sentiment.py`:

```python
EMOJI_SENTIMENT = {
    '😊': 0.8,
    '😢': -0.8,
    # Add your custom emoji mappings
    '🤖': 0.5,
    '🔥': 0.7
}
```

### Custom Multimodal Weights

Modify fusion weights in `analyze_multimodal()`:

```python
# Default: 70% text, 30% emoji
text_weight = 0.7
emoji_weight = 0.3

# For emoji-heavy chats: 50% text, 50% emoji
text_weight = 0.5
emoji_weight = 0.5

combined_sentiment = (
    text_weight * text_sentiment +
    emoji_weight * emoji_sentiment
)
```

### Adding New Language Models

Add to `MULTILINGUAL_MODELS` dictionary:

```python
MULTILINGUAL_MODELS = {
    'your_language': {
        'sentiment': 'huggingface-model-id-sentiment',
        'emotion': 'huggingface-model-id-emotion'
    }
}
```

## Citation

If you use this system in research:

### Multimodal Sentiment Analysis
```
@software{whatsapp_multimodal_analyzer,
  title = {WhatsApp Multimodal Sentiment Analyzer},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/whatsapp_sentiment_analyzer}
}
```

### Underlying Models
- Language detection: https://github.com/Mimino666/langdetect
- Multilingual BERT: https://github.com/google-research/bert
- RoBERTa sentiment: See ADVANCED_SENTIMENT_GUIDE.md

## Resources

- [HuggingFace Models](https://huggingface.co/models)
- [Langdetect Documentation](https://pypi.org/project/langdetect/)
- [Emoji Package](https://pypi.org/project/emoji/)
- [Example Scripts](./example_multimodal.py)
- [Advanced Sentiment Guide](./ADVANCED_SENTIMENT_GUIDE.md)

## Support

For issues:
1. Check this guide
2. Review example scripts
3. Check troubleshooting section
4. Open GitHub issue

---

**Version**: 2.0
**Last Updated**: 2024
**License**: MIT
