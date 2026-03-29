"""
Enhanced Multimodal and Multilingual Sentiment Analysis

This module extends the advanced sentiment analysis with:
- Multilingual support (100+ languages)
- Language detection
- Multimodal analysis (text + emoji)
- Enhanced preprocessing for transformers
- Multiple emotion detection models
- Robust error handling
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import Counter

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

# Import language detection
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Install with: pip install langdetect")

# Emoji handling
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    logger.info("emoji library not available. Emoji sentiment will be limited.")


class MultimodalSentimentAnalyzer:
    """
    Advanced multimodal and multilingual sentiment analyzer.

    Features:
    - Multilingual sentiment analysis (100+ languages)
    - Language detection
    - Emoji sentiment analysis
    - Combined text + emoji analysis
    - Multiple emotion models
    - Confidence scores
    """

    # Predefined model configurations for different languages
    MULTILINGUAL_MODELS = {
        'english': {
            'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'emotion': 'j-hartmann/emotion-english-distilroberta-base'
        },
        'multilingual': {
            'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment',
            'emotion': 'joeddav/xlm-roberta-large-xnli'
        },
        'spanish': {
            'sentiment': 'finiteautomata/beto-sentiment-analysis',
            'emotion': None
        },
        'french': {
            'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment',
            'emotion': None
        },
        'german': {
            'sentiment': 'oliverguhr/german-sentiment-bert',
            'emotion': None
        },
        'arabic': {
            'sentiment': 'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment',
            'emotion': None
        },
        'chinese': {
            'sentiment': 'uer/roberta-base-finetuned-jd-binary-chinese',
            'emotion': None
        },
        'hindi': {
            'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment',
            'emotion': None
        }
    }

    # Emoji sentiment dictionary (common emojis with sentiment polarity)
    EMOJI_SENTIMENT = {
        '😊': 0.8, '😃': 0.9, '😄': 0.9, '😁': 0.8, '😆': 0.7,
        '😍': 0.95, '🥰': 0.95, '😘': 0.85, '💕': 0.9, '❤️': 0.9,
        '😂': 0.7, '🤣': 0.8, '😅': 0.5, '😉': 0.6, '😌': 0.6,
        '🎉': 0.85, '🎊': 0.85, '🎈': 0.7, '💪': 0.75, '👍': 0.8,
        '🙏': 0.6, '🤗': 0.8, '🥳': 0.9, '✨': 0.7, '🌟': 0.75,
        '😢': -0.8, '😭': -0.9, '😞': -0.7, '😔': -0.7, '😟': -0.6,
        '😠': -0.8, '😡': -0.9, '🤬': -0.95, '😤': -0.7, '💔': -0.85,
        '😰': -0.7, '😨': -0.8, '😱': -0.9, '😖': -0.7, '😣': -0.7,
        '🙄': -0.4, '😒': -0.5, '😑': -0.3, '😐': 0.0, '😶': 0.0,
        '🤔': 0.1, '🤨': -0.2, '😬': -0.3, '🤐': -0.2, '😕': -0.4
    }

    def __init__(
        self,
        language: str = 'english',
        sentiment_model: Optional[str] = None,
        emotion_model: Optional[str] = None,
        device: Optional[str] = None,
        enable_emoji_analysis: bool = True,
        enable_multimodal: bool = True,
        include_emotion: bool = False
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required. Install with: pip install transformers torch")

        self.language = language.lower()
        self.enable_emoji_analysis = enable_emoji_analysis
        self.enable_multimodal = enable_multimodal
        self.include_emotion = include_emotion

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1

        if sentiment_model:
            self.sentiment_model_name = sentiment_model
        elif self.language in self.MULTILINGUAL_MODELS:
            self.sentiment_model_name = self.MULTILINGUAL_MODELS[self.language]['sentiment']
        else:
            logger.warning(f"Language '{self.language}' not found, using multilingual model")
            self.sentiment_model_name = self.MULTILINGUAL_MODELS['multilingual']['sentiment']

        if emotion_model:
            self.emotion_model_name = emotion_model
        elif self.language in self.MULTILINGUAL_MODELS:
            self.emotion_model_name = self.MULTILINGUAL_MODELS[self.language].get('emotion')
        else:
            self.emotion_model_name = None

        self._init_pipelines()

    def _init_pipelines(self):
        try:
            logger.info(f"Loading sentiment model: {self.sentiment_model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                device=self.device,
                truncation=True,
                max_length=512,
                top_k=None # Get scores for all classes
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            self.sentiment_pipeline = None

        if self.emotion_model_name and self.include_emotion:
            try:
                logger.info(f"Loading emotion model: {self.emotion_model_name}")
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model=self.emotion_model_name,
                    device=self.device,
                    truncation=True,
                    max_length=512,
                    top_k=None
                )
                logger.info("Emotion model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading emotion model: {e}")
                self.emotion_pipeline = None
        else:
            logger.info(f"Emotion model not loaded for language: {self.language} (or not requested).")
            self.emotion_pipeline = None

    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'https?://\S+', '', text)
        text = text.strip()
        return text

    def analyze_emoji_sentiment(self, emojis: List[str]) -> Dict[str, float]:
        if not emojis:
            return {'emoji_sentiment': 0.0, 'emoji_count': 0, 'positive_emojis': 0, 'negative_emojis': 0, 'neutral_emojis': 0}

        sentiment_scores = [self.EMOJI_SENTIMENT[e] for e in emojis if e in self.EMOJI_SENTIMENT]
        if not sentiment_scores:
            avg_sentiment = 0.0
        else:
            avg_sentiment = np.mean(sentiment_scores)

        positive = sum(1 for s in sentiment_scores if s > 0.2)
        negative = sum(1 for s in sentiment_scores if s < -0.2)
        neutral = len(sentiment_scores) - positive - negative

        return {
            'emoji_sentiment': float(avg_sentiment),
            'emoji_count': len(emojis),
            'positive_emojis': positive,
            'negative_emojis': negative,
            'neutral_emojis': neutral
        }

    def _get_continuous_sentiment_score(self, text: str) -> dict:
        if not self.sentiment_pipeline:
            return {'label': 'unknown', 'score': 0.0, 'confidence': 0.0}

        if not text or len(text.strip()) == 0:
            return {'label': 'neutral', 'score': 0.0, 'confidence': 1.0}

        try:
            raw_outputs = self.sentiment_pipeline(text[:512])[0]
            scores = {item['label'].lower(): item['score'] for item in raw_outputs}

            label_map = {
                "label_0": "negative", "label_1": "neutral", "label_2": "positive",
                "1 star": "negative", "2 stars": "negative", "3 stars": "neutral",
                "4 stars": "positive", "5 stars": "positive"
            }
            
            mapped_scores = {}
            for label, score in scores.items():
                mapped_label = label_map.get(label, label)
                mapped_scores[mapped_label] = mapped_scores.get(mapped_label, 0) + score

            pos_score = mapped_scores.get('positive', 0.0)
            neg_score = mapped_scores.get('negative', 0.0)
            
            continuous_score = pos_score - neg_score
            
            final_label = max(mapped_scores, key=mapped_scores.get)
            confidence = mapped_scores[final_label]
            
            return {'label': final_label, 'score': continuous_score, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Error getting continuous score: {e}")
            return {'label': 'unknown', 'score': 0.0, 'confidence': 0.0}

    def analyze_multimodal(self, text: str, emojis: List[str]) -> Dict[str, any]:
        # Analyze text sentiment (including emojis for the model to see)
        preprocessed_text = self.preprocess_text(text)
        text_result = self._get_continuous_sentiment_score(preprocessed_text)
        text_sentiment = text_result.get('score', 0.0)

        # Analyze emoji sentiment
        emoji_result = self.analyze_emoji_sentiment(emojis) if self.enable_emoji_analysis else {}
        emoji_sentiment = emoji_result.get('emoji_sentiment', 0.0)
        
        # Fuse scores if multimodal is enabled and emojis are present
        if self.enable_multimodal and emojis and self.enable_emoji_analysis:
            # If text is empty or just whitespace, emoji sentiment is the only sentiment
            if not preprocessed_text:
                combined_sentiment = emoji_sentiment
            else:
                text_weight = 0.7
                emoji_weight = 0.3
                combined_sentiment = (text_weight * text_sentiment) + (emoji_weight * emoji_sentiment)
            
            # Clamp the score to be within [-1, 1]
            combined_sentiment = max(-1.0, min(1.0, combined_sentiment))

            if combined_sentiment > 0.15:
                final_label = 'positive'
            elif combined_sentiment < -0.15:
                final_label = 'negative'
            else:
                final_label = 'neutral'
            
            return {
                'mm_sentiment': final_label,
                'mm_sentiment_score': combined_sentiment,
                'mm_text_sentiment': text_sentiment,
                'mm_emoji_sentiment': emoji_sentiment,
                'mm_confidence': text_result.get('confidence', 0.0),
                'is_multimodal': True,
                **emoji_result,
            }
        # Fallback for messages with no emojis or when multimodal is disabled
        else:
            return {
                'mm_sentiment': text_result.get('label', 'neutral'),
                'mm_sentiment_score': text_sentiment,
                'mm_text_sentiment': text_sentiment,
                'mm_emoji_sentiment': 0.0,
                'mm_confidence': text_result.get('confidence', 0.0),
                'is_multimodal': False,
                **emoji_result,
            }

    def analyze_emotion(self, text: str) -> Dict[str, any]:
        if not self.emotion_pipeline:
            return {"emotion": "unknown", "score": 0.0, "all_emotions": {}}
        
        preprocessed = self.preprocess_text(text)
        if not preprocessed:
            return {"emotion": "neutral", "score": 1.0, "all_emotions": {"neutral": 1.0}}

        try:
            results = self.emotion_pipeline(preprocessed[:512])[0]
            all_emotions = {item['label']: float(item['score']) for item in results}
            primary = max(results, key=lambda x: x['score'])
            return {"emotion": primary['label'], "score": float(primary['score']), "all_emotions": all_emotions}
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {"emotion": "unknown", "score": 0.0, "all_emotions": {}}

def detect_language(text: str) -> Dict[str, any]:
    if not LANGDETECT_AVAILABLE:
        return {'language': 'unknown', 'confidence': 0.0, 'all_languages': []}
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return {'language': 'unknown', 'confidence': 0.0, 'all_languages': []}
    try:
        lang_probs = detect_langs(text)
        primary_lang = lang_probs[0]
        return {
            'language': primary_lang.lang,
            'confidence': primary_lang.prob,
            'all_languages': [{'lang': lp.lang, 'prob': lp.prob} for lp in lang_probs[:3]]
        }
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        return {'language': 'unknown', 'confidence': 0.0, 'all_languages': []}

def apply_multimodal_sentiment(
    df: pd.DataFrame,
    text_column: str = 'message',
    emoji_column: str = 'emojis',
    language: str = 'english',
    sentiment_model: Optional[str] = None,
    emotion_model: Optional[str] = None,
    enable_language_detection: bool = False,
    enable_emoji_analysis: bool = True,
    enable_multimodal: bool = True,
    include_emotion: bool = False
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    if emoji_column not in df.columns and enable_emoji_analysis:
        # If emoji analysis is on, the column must exist.
        raise ValueError(f"Column '{emoji_column}' not found. Please run preprocess_df first.")

    df = df.copy()
    
    # Ensure emoji column exists and is not all NaN, if not, create it as empty lists
    if emoji_column not in df.columns:
        df[emoji_column] = [[] for _ in range(len(df))]
    else:
        # Fill any NaN values with empty lists to prevent errors
        df[emoji_column] = df[emoji_column].apply(lambda x: x if isinstance(x, list) else [])


    if enable_language_detection and LANGDETECT_AVAILABLE:
        logger.info("Detecting languages...")
        texts = df[text_column].fillna('').astype(str).tolist()
        lang_results = [detect_language(text) for text in texts]
        df['detected_language'] = [r['language'] for r in lang_results]
        df['language_confidence'] = [r['confidence'] for r in lang_results]

    try:
        analyzer = MultimodalSentimentAnalyzer(
            language=language,
            sentiment_model=sentiment_model,
            emotion_model=emotion_model,
            enable_emoji_analysis=enable_emoji_analysis,
            enable_multimodal=enable_multimodal,
            include_emotion=include_emotion
        )
    except ImportError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise

    logger.info(f"Analyzing {len(df)} messages with multimodal analysis...")
    
    results = [
        analyzer.analyze_multimodal(row[text_column], row[emoji_column])
        for _, row in df.fillna({text_column: ''}).iterrows()
    ]
    
    df['mm_sentiment'] = [r.get('mm_sentiment', 'neutral') for r in results]
    df['mm_sentiment_score'] = [r.get('mm_sentiment_score', 0.0) for r in results]
    df['mm_text_sentiment'] = [r.get('mm_text_sentiment', 0.0) for r in results]
    df['mm_confidence'] = [r.get('mm_confidence', 0.0) for r in results]

    if enable_emoji_analysis:
        df['mm_emoji_sentiment'] = [r.get('mm_emoji_sentiment', 0.0) for r in results]
        df['mm_emoji_count'] = [r.get('emoji_count', 0) for r in results]

    if analyzer.emotion_pipeline:
        emotion_results = [analyzer.analyze_emotion(text) for text in df[text_column].fillna('').astype(str)]
        df['mm_emotion'] = [r['emotion'] for r in emotion_results]
        df['mm_emotion_score'] = [r['score'] for r in emotion_results]
        df['mm_all_emotions'] = [r['all_emotions'] for r in emotion_results]

    logger.info("Multimodal analysis complete!")
    return df

SUPPORTED_LANGUAGES = list(MultimodalSentimentAnalyzer.MULTILINGUAL_MODELS.keys())

def get_available_models(language: str = 'all') -> Dict[str, Dict]:
    if language == 'all':
        return MultimodalSentimentAnalyzer.MULTILINGUAL_MODELS
    elif language in MultimodalSentimentAnalyzer.MULTILINGUAL_MODELS:
        return {language: MultimodalSentimentAnalyzer.MULTILINGUAL_MODELS[language]}
    else:
        return {}