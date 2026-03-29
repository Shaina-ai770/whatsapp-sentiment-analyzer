# Advanced Sentiment Analysis Guide

## Overview

This guide explains how to use the advanced sentiment analysis features powered by transformer-based models (BERT/RoBERTa) in the WhatsApp Sentiment Analyzer project.

## Features

### 1. Advanced Sentiment Analysis
- Uses state-of-the-art transformer models from HuggingFace
- More accurate than traditional VADER sentiment analysis
- Provides confidence scores for each prediction
- Supports multiple pre-trained models

### 2. Emotion Detection
- Detects specific emotions beyond basic sentiment
- Supports emotions: joy, sadness, anger, fear, surprise, love
- Provides confidence scores for all emotions
- Identifies primary emotion with highest confidence

### 3. Streamlit Integration
- Toggle between VADER and transformer-based analysis
- Interactive emotion filtering
- Model comparison visualizations
- Confidence score analysis

## Installation

### Requirements

Install the additional dependencies for advanced sentiment analysis:

```bash
pip install transformers>=4.30.0 torch>=2.0.0 sentencepiece accelerate
```

Or install all requirements at once:

```bash
pip install -r requirements.txt
```

### First Run

On first run, the models will be downloaded (~500MB). This may take a few minutes depending on your internet connection. Subsequent runs will be much faster as models are cached locally.

## Usage

### 1. Using in Python Scripts

#### Quick Analysis of a Single Message

```python
from src.advanced_sentiment import quick_analyze

text = "I'm so excited about this project! 🎉"
result = quick_analyze(text, include_emotion=True)

print(f"Sentiment: {result['sentiment']['label']}")
print(f"Confidence: {result['sentiment']['score']:.2%}")
print(f"Emotion: {result['emotion']['emotion']}")
```

#### Batch Analysis of WhatsApp Chat

```python
from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.advanced_sentiment import apply_advanced_sentiment

# Parse and preprocess chat
with open('chat.txt', 'r') as f:
    chat_data = f.read()

df = parse_chat(chat_data)
df = preprocess_df(df)

# Apply advanced sentiment analysis
df = apply_advanced_sentiment(df, include_emotions=True)

# Access results
print(df[['message', 'transformer_sentiment', 'transformer_emotion']].head())
```

#### Using Custom Models

```python
from src.advanced_sentiment import AdvancedSentimentAnalyzer

# Initialize with custom models
analyzer = AdvancedSentimentAnalyzer(
    sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
    emotion_model="bhadresh-savani/distilbert-base-uncased-emotion"
)

# Analyze sentiment
sentiment = analyzer.analyze_sentiment("This is amazing!")
print(sentiment)

# Analyze emotion
emotion = analyzer.analyze_emotion("I'm feeling worried about this")
print(emotion)
```

### 2. Using the Streamlit App

1. **Launch the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Select Analysis Method:**
   - In the sidebar, choose "Transformers (Advanced)" under "Sentiment Analysis Method"
   - Optionally enable "Include Emotion Detection"

3. **Upload Chat File:**
   - Upload your WhatsApp chat export (.txt file)
   - The app will automatically apply advanced analysis

4. **Explore Results:**
   - **Dashboard Tab:** View sentiment and emotion distributions
   - **Chat Table Tab:** See detailed results for each message
   - **Per-User Analysis Tab:** Compare sentiment/emotion by user
   - **Advanced Analytics Tab:**
     - Model comparison (VADER vs Transformers)
     - Confidence score analysis
     - Detailed emotion breakdown
     - High-confidence prediction samples

### 3. Running Examples

Run the comprehensive example script:

```bash
python example_advanced_sentiment.py
```

This demonstrates:
- Quick analysis of individual messages
- Batch analysis with sample chat data
- Per-user sentiment and emotion analysis
- Using custom models
- Error handling for edge cases

## API Reference

### AdvancedSentimentAnalyzer

Main class for sentiment and emotion analysis.

```python
class AdvancedSentimentAnalyzer:
    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[str] = None
    )
```

**Methods:**

- `analyze_sentiment(text: str) -> Dict`: Analyze sentiment of a single text
- `analyze_emotion(text: str) -> Dict`: Analyze emotions in a single text
- `analyze_batch(texts: List[str]) -> Tuple`: Analyze multiple texts efficiently

### apply_advanced_sentiment

Apply transformer-based sentiment analysis to a DataFrame.

```python
def apply_advanced_sentiment(
    df: pd.DataFrame,
    text_column: str = 'message',
    sentiment_model: Optional[str] = None,
    emotion_model: Optional[str] = None,
    include_emotions: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `df`: Input DataFrame with text messages
- `text_column`: Name of column containing text (default: 'message')
- `sentiment_model`: Optional custom sentiment model ID
- `emotion_model`: Optional custom emotion model ID
- `include_emotions`: Whether to include emotion detection

**Returns:**
DataFrame with additional columns:
- `transformer_sentiment`: Sentiment label (positive/negative/neutral)
- `transformer_sentiment_score`: Confidence score (0-1)
- `transformer_emotion`: Primary emotion detected
- `transformer_emotion_score`: Emotion confidence score
- `transformer_all_emotions`: Dictionary of all emotion scores

### compare_sentiments

Compare VADER sentiment with transformer-based sentiment.

```python
def compare_sentiments(df: pd.DataFrame) -> pd.DataFrame
```

Returns a summary DataFrame showing agreement rates between the two methods.

### quick_analyze

Quick analysis of a single text message.

```python
def quick_analyze(text: str, include_emotion: bool = True) -> Dict
```

## Available Pre-trained Models

### Sentiment Analysis Models

1. **cardiffnlp/twitter-roberta-base-sentiment-latest** (Default)
   - Best for social media and informal text
   - Trained on Twitter data
   - Excellent emoji and slang support

2. **distilbert-base-uncased-finetuned-sst-2-english**
   - Faster, lighter model
   - Good for general text
   - Binary sentiment (positive/negative)

3. **nlptown/bert-base-multilingual-uncased-sentiment**
   - Multilingual support
   - 5-star rating scale output

### Emotion Detection Models

1. **j-hartmann/emotion-english-distilroberta-base** (Default)
   - 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
   - High accuracy on English text
   - Good balance of speed and performance

2. **bhadresh-savani/distilbert-base-uncased-emotion**
   - 6 emotions: sadness, joy, love, anger, fear, surprise
   - Faster inference
   - Good for real-time applications

3. **SamLowe/roberta-base-go_emotions**
   - 28 fine-grained emotions
   - Very detailed emotion analysis
   - Slower but more comprehensive

## Performance Considerations

### Speed
- **VADER:** ~1000 messages/second (very fast)
- **Transformers:** ~10-50 messages/second (depends on hardware)
- **First run:** Slower due to model downloads

### Accuracy
- **VADER:** Good for general sentiment, simple lexicon-based
- **Transformers:** State-of-the-art accuracy, context-aware

### Hardware
- **CPU:** Works fine, but slower
- **GPU:** Significantly faster (10-50x speedup)
- **Memory:** Models require ~500MB-1GB RAM each

### Recommendations
- Use **VADER** for quick exploration or large datasets (>10k messages)
- Use **Transformers** for accurate analysis of important conversations
- Use **GPU** if available for large-scale transformer analysis

## Comparison: VADER vs Transformers

| Feature | VADER | Transformers |
|---------|-------|--------------|
| **Speed** | Very Fast | Moderate |
| **Accuracy** | Good | Excellent |
| **Context Awareness** | Limited | Strong |
| **Emoji Support** | Basic | Excellent |
| **Emotion Detection** | No | Yes |
| **Setup** | Simple | Requires model download |
| **Dependencies** | Lightweight | Heavy (PyTorch) |
| **Best For** | Quick analysis, exploration | Accurate analysis, research |

## Troubleshooting

### Models Not Downloading
- Check internet connection
- Try manually downloading: `python -c "from transformers import pipeline; pipeline('sentiment-analysis')"`
- Clear cache: `rm -rf ~/.cache/huggingface/`

### Out of Memory Errors
- Reduce batch size in code
- Use smaller model (e.g., DistilBERT instead of RoBERTa)
- Close other applications
- Use CPU instead of GPU (slower but uses less memory)

### Slow Performance
- First run is always slow (downloading models)
- Use GPU if available
- Reduce dataset size for testing
- Consider using VADER for large datasets

### Import Errors
```bash
# If you get ImportError for transformers
pip install transformers torch

# If you get ImportError for sentencepiece
pip install sentencepiece

# If you get ImportError for accelerate
pip install accelerate
```

## Examples

### Example 1: Compare VADER vs Transformer Results

```python
from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.sentiment import apply_vader
from src.advanced_sentiment import apply_advanced_sentiment, compare_sentiments

# Load and process data
with open('chat.txt', 'r') as f:
    df = parse_chat(f.read())
df = preprocess_df(df)

# Apply both methods
df = apply_vader(df)
df = apply_advanced_sentiment(df)

# Compare results
comparison = compare_sentiments(df)
print(comparison)

# Find disagreements
disagreements = df[df['sentiment'] != df['transformer_sentiment']]
print(f"\nDisagreements: {len(disagreements)}")
print(disagreements[['message', 'sentiment', 'transformer_sentiment']].head())
```

### Example 2: Emotion-Based Filtering

```python
from src.advanced_sentiment import apply_advanced_sentiment

# Apply analysis with emotions
df = apply_advanced_sentiment(df, include_emotions=True)

# Filter by emotion
joy_messages = df[df['transformer_emotion'] == 'joy']
anger_messages = df[df['transformer_emotion'] == 'anger']

print(f"Joyful messages: {len(joy_messages)}")
print(f"Angry messages: {len(anger_messages)}")

# Show high-confidence emotional messages
high_joy = joy_messages.nlargest(5, 'transformer_emotion_score')
print("\nMost joyful messages:")
print(high_joy[['message', 'transformer_emotion_score']])
```

### Example 3: Per-User Emotion Analysis

```python
# Emotion distribution by user
emotion_by_user = df.groupby('author')['transformer_emotion'].value_counts()
print(emotion_by_user)

# Average emotion scores by user
for emotion in ['joy', 'sadness', 'anger']:
    df[f'{emotion}_score'] = df['transformer_all_emotions'].apply(
        lambda x: x.get(emotion, 0) if isinstance(x, dict) else 0
    )

emotion_summary = df.groupby('author')[['joy_score', 'sadness_score', 'anger_score']].mean()
print(emotion_summary)
```

## Best Practices

1. **Start with VADER:** Use VADER first to explore data quickly
2. **Use Transformers for Deep Analysis:** Apply transformer models when accuracy matters
3. **Cache Results:** Save analyzed DataFrames to avoid reprocessing
4. **Filter Before Analysis:** Reduce data size before transformer analysis if possible
5. **Monitor Confidence:** Low confidence scores indicate uncertain predictions
6. **Validate Results:** Manually check samples to ensure model is working correctly
7. **Consider Context:** Remember that sentiment/emotion depends on context and may be ambiguous

## Citation

If you use these models in research, please cite:

### RoBERTa Sentiment Model
```
@inproceedings{barbieri-etal-2020-tweeteval,
    title = "{T}weet{E}val: Unified Benchmark and Comparative Evaluation for Tweet Classification",
    author = "Barbieri, Francesco and Camacho-Collados, Jose and Espinosa Anke, Luis and Neves, Leonardo",
    booktitle = "Findings of EMNLP",
    year = "2020"
}
```

### Emotion Detection Model
```
@article{hartmann2022more,
  title={More than a Feeling: Accuracy and Application of Sentiment Analysis},
  author={Hartmann, Jochen and Heitmann, Mark and Siebert, Christian and Schamp, Christina},
  journal={International Journal of Research in Marketing},
  year={2022}
}
```

## Further Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Project GitHub Repository](https://github.com/yourusername/whatsapp_sentiment_analyzer)

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review example scripts in `example_advanced_sentiment.py`
3. Open an issue on GitHub
4. Check HuggingFace documentation for model-specific issues
