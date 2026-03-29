"""
Advanced Sentiment Analysis using Transformer Models (BERT/RoBERTa)

This module provides state-of-the-art sentiment and emotion analysis using
pre-trained transformer models from HuggingFace. It offers more accurate
sentiment detection and emotion classification compared to traditional methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import transformers with error handling
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Install with: pip install transformers torch")


class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer using transformer-based models.

    Supports:
    - Sentiment analysis (positive, negative, neutral)
    - Emotion detection (joy, sadness, anger, fear, surprise, love)
    - Confidence scores for predictions
    """

    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[str] = None
    ):
        """
        Initialize the advanced sentiment analyzer.

        Args:
            sentiment_model: HuggingFace model ID for sentiment analysis
            emotion_model: HuggingFace model ID for emotion detection
            device: Device to run models on ('cpu', 'cuda', or None for auto-detect)

        Raises:
            ImportError: If transformers library is not installed
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required. Install with: "
                "pip install transformers torch"
            )

        # Auto-detect device if not specified
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1

        self.sentiment_model_name = sentiment_model
        self.emotion_model_name = emotion_model

        # Initialize pipelines
        self._init_pipelines()

    def _init_pipelines(self):
        """Initialize the sentiment and emotion analysis pipelines."""
        try:
            logger.info(f"Loading sentiment model: {self.sentiment_model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                device=self.device,
                truncation=True,
                max_length=512
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            self.sentiment_pipeline = None

        try:
            logger.info(f"Loading emotion model: {self.emotion_model_name}")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                device=self.device,
                truncation=True,
                max_length=512,
                top_k=None  # Return all emotion scores
            )
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            self.emotion_pipeline = None

    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single text message.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing:
                - label: Sentiment label (positive, negative, neutral)
                - score: Confidence score (0-1)
                - raw_label: Original model output label
        """
        if not self.sentiment_pipeline:
            return {"label": "unknown", "score": 0.0, "raw_label": "unknown"}

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return {"label": "neutral", "score": 1.0, "raw_label": "neutral"}

        try:
            result = self.sentiment_pipeline(text[:512])[0]
            label_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }
            mapped_label = label_map.get(result['label'], result['label'])

            return {
                "label": mapped_label,
                "score": float(result['score']),
                "raw_label": result['label']
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"label": "unknown", "score": 0.0, "raw_label": "unknown"}

    def analyze_emotion(self, text: str) -> Dict[str, any]:
        """
        Analyze emotions in a single text message.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing:
                - emotion: Primary emotion detected
                - score: Confidence score for primary emotion (0-1)
                - all_emotions: Dictionary with scores for all emotions
        """
        if not self.emotion_pipeline:
            return {
                "emotion": "unknown",
                "score": 0.0,
                "all_emotions": {}
            }

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "emotion": "neutral",
                "score": 1.0,
                "all_emotions": {"neutral": 1.0}
            }

        try:
            results = self.emotion_pipeline(text[:512])[0]

            # Extract all emotion scores
            all_emotions = {item['label']: float(item['score']) for item in results}

            # Get primary emotion (highest score)
            primary = max(results, key=lambda x: x['score'])

            return {
                "emotion": primary['label'],
                "score": float(primary['score']),
                "all_emotions": all_emotions
            }
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {
                "emotion": "unknown",
                "score": 0.0,
                "all_emotions": {}
            }

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze sentiment and emotion for a batch of texts efficiently.

        Args:
            texts: List of text messages to analyze
            batch_size: Number of texts to process at once

        Returns:
            Tuple of (sentiment_results, emotion_results)
        """
        sentiment_results = []
        emotion_results = []

        for text in texts:
            sentiment_results.append(self.analyze_sentiment(text))
            emotion_results.append(self.analyze_emotion(text))

        return sentiment_results, emotion_results


def apply_advanced_sentiment(
    df: pd.DataFrame,
    text_column: str = 'message',
    sentiment_model: Optional[str] = None,
    emotion_model: Optional[str] = None,
    include_emotions: bool = True
) -> pd.DataFrame:
    """
    Apply advanced transformer-based sentiment analysis to a DataFrame.

    This function adds sentiment and emotion analysis columns to the input DataFrame
    using state-of-the-art transformer models (BERT/RoBERTa).

    Args:
        df: Input DataFrame with text messages
        text_column: Name of column containing text to analyze (default: 'message')
        sentiment_model: Optional custom sentiment model ID from HuggingFace
        emotion_model: Optional custom emotion model ID from HuggingFace
        include_emotions: Whether to include emotion detection (default: True)

    Returns:
        pd.DataFrame: DataFrame with additional columns:
            - transformer_sentiment: Sentiment label (positive/negative/neutral)
            - transformer_sentiment_score: Confidence score (0-1)
            - transformer_emotion: Primary emotion detected (if include_emotions=True)
            - transformer_emotion_score: Emotion confidence score
            - transformer_all_emotions: Dictionary of all emotion scores

    Example:
        >>> df = preprocess_df(parse_chat(chat_content))
        >>> df = apply_advanced_sentiment(df)
        >>> print(df[['message', 'transformer_sentiment', 'transformer_emotion']].head())

    Raises:
        ValueError: If text_column doesn't exist in DataFrame
        ImportError: If transformers library is not installed
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    df = df.copy()

    # Initialize analyzer with custom models if provided
    kwargs = {}
    if sentiment_model:
        kwargs['sentiment_model'] = sentiment_model
    if emotion_model:
        kwargs['emotion_model'] = emotion_model

    try:
        analyzer = AdvancedSentimentAnalyzer(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise

    # Prepare texts for analysis
    texts = df[text_column].fillna('').astype(str).tolist()

    logger.info(f"Analyzing {len(texts)} messages with transformer models...")

    # Perform batch analysis
    sentiment_results, emotion_results = analyzer.analyze_batch(texts)

    # Add sentiment columns
    df['transformer_sentiment'] = [r['label'] for r in sentiment_results]
    df['transformer_sentiment_score'] = [r['score'] for r in sentiment_results]

    # Add emotion columns if requested
    if include_emotions:
        df['transformer_emotion'] = [r['emotion'] for r in emotion_results]
        df['transformer_emotion_score'] = [r['score'] for r in emotion_results]
        df['transformer_all_emotions'] = [r['all_emotions'] for r in emotion_results]

    logger.info("Analysis complete!")

    return df


def compare_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare VADER sentiment with transformer-based sentiment.

    Useful for understanding differences between traditional and advanced methods.

    Args:
        df: DataFrame with both 'sentiment' (VADER) and 'transformer_sentiment' columns

    Returns:
        pd.DataFrame: Comparison summary showing agreement rates
    """
    if 'sentiment' not in df.columns or 'transformer_sentiment' not in df.columns:
        raise ValueError("DataFrame must contain both 'sentiment' and 'transformer_sentiment' columns")

    comparison = pd.DataFrame({
        'VADER': df['sentiment'],
        'Transformer': df['transformer_sentiment'],
        'Agreement': df['sentiment'] == df['transformer_sentiment']
    })

    agreement_rate = comparison['Agreement'].mean()

    summary = pd.DataFrame({
        'Metric': ['Agreement Rate', 'Total Messages', 'Agreements', 'Disagreements'],
        'Value': [
            f"{agreement_rate:.2%}",
            len(comparison),
            comparison['Agreement'].sum(),
            (~comparison['Agreement']).sum()
        ]
    })

    return summary


# Convenience function for quick analysis
def quick_analyze(text: str, include_emotion: bool = True) -> Dict:
    """
    Quick analysis of a single text message.

    Args:
        text: Text to analyze
        include_emotion: Whether to include emotion detection

    Returns:
        Dictionary with sentiment and optionally emotion results
    """
    analyzer = AdvancedSentimentAnalyzer()

    result = {
        "text": text,
        "sentiment": analyzer.analyze_sentiment(text)
    }

    if include_emotion:
        result["emotion"] = analyzer.analyze_emotion(text)

    return result
