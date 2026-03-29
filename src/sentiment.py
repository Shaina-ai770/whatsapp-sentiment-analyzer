import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def apply_vader(df):
    """Apply VADER sentiment analysis for text polarity and intensity.

    This function uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
    a lexicon-based, rule-based sentiment analysis tool. It is strictly
    limited to analyzing the polarity (positive, negative, neutral) of text content.

    IMPORTANT:
    - VADER does NOT support emotion detection.
    - This implementation leverages 'is_system_message' and 'is_emoji_only' flags
      from preprocessing to assign neutral sentiment to such messages.

    The function calculates sentiment scores and classifies each message into five
    intensity levels based on the compound score.

    Args:
        df (pd.DataFrame): DataFrame with 'message' column and 'is_system_message',
                           'is_emoji_only' boolean flags from preprocessing.

    Returns:
        pd.DataFrame: DataFrame with additional sentiment columns:
            - vader_sentiment: Categorical label ('positive', 'negative', 'neutral').
            - sentiment_intensity: Detailed intensity level.
            - vader_pos: Positive score component.
            - vader_neu: Neutral score component.
            - vader_neg: Negative score component.
            - vader_compound: Compound sentiment score from -1 to +1.

    Sentiment Classification Thresholds:
        - Very Positive: compound >= 0.5
        - Positive: 0.05 <= compound < 0.5
        - Neutral: -0.05 < compound < 0.05
        - Negative: -0.5 < compound <= -0.05
        - Very Negative: compound <= -0.5
    """
    df = df.copy()

    def get_sentiment(row):
        # Prioritize system messages: they have no sentiment
        if row['is_system_message']:
            return {
                'vader_pos': 0.0,
                'vader_neu': 0.0, # System messages have no 'neutral' textual content
                'vader_neg': 0.0,
                'vader_compound': 0.0,
                'sentiment': 'neutral',
                'sentiment_intensity': 'Neutral'
            }
        
        # Then handle emoji-only messages: always neutral with full neutral score
        if row['is_emoji_only']:
            return {
                'vader_pos': 0.0,
                'vader_neu': 1.0, # Considered entirely neutral by VADER
                'vader_neg': 0.0,
                'vader_compound': 0.0,
                'sentiment': 'neutral',
                'sentiment_intensity': 'Neutral'
            }

        message_text = row['message']
        
        # If message is effectively empty after cleaning (e.g., only URLs, system message, or just whitespace)
        if not message_text.strip():
             return {
                'vader_pos': 0.0,
                'vader_neu': 1.0,
                'vader_neg': 0.0,
                'vader_compound': 0.0,
                'sentiment': 'neutral',
                'sentiment_intensity': 'Neutral'
            }

        scores = analyzer.polarity_scores(message_text)
        compound = scores.get('compound', 0.0)

        # Apply official VADER sentiment thresholds
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Intensity level
        if compound >= 0.5:
            intensity = 'Very Positive'
        elif compound >= 0.05:
            intensity = 'Positive'
        elif compound > -0.05:
            intensity = 'Neutral'
        elif compound > -0.5:
            intensity = 'Negative'
        else:
            intensity = 'Very Negative'

        return {
            'vader_pos': scores.get('pos', 0.0),
            'vader_neu': scores.get('neu', 0.0),
            'vader_neg': scores.get('neg', 0.0),
            'vader_compound': compound,
            'sentiment': sentiment,
            'sentiment_intensity': intensity
        }

    sentiment_data = df.apply(get_sentiment, axis=1, result_type='expand')
    df = pd.concat([df, sentiment_data], axis=1)

    return df


def per_user_summary(df, by=['author', 'mobile']):
    """Return per-user sentiment summary grouped by the specified columns.

    This function aggregates sentiment data to provide a per-user summary,
    including message counts, sentiment distribution percentages, and average
    compound score.

    Args:
        df (pd.DataFrame): DataFrame with sentiment columns from apply_vader.
        by (list): List of columns to group by (e.g., ['author']).

    Returns:
        pd.DataFrame: A summary DataFrame with one row per user, containing
                      aggregated sentiment metrics.
    """
    group_cols = by if isinstance(by, list) else [by]
    df_copy = df.copy()
    
    # Ensure sentiment and compound exist
    if 'sentiment' not in df_copy.columns or 'vader_compound' not in df_copy.columns:
        # If sentiment not applied, apply it. This will now respect system/emoji-only messages.
        df_copy = apply_vader(df_copy)

    # Ensure group columns exist in DataFrame
    for col in group_cols:
        if col not in df_copy.columns:
            df_copy[col] = "Unknown" # Assign a default value if missing

    # Use dropna=False so groups with NaN (None) are included
    message_count = df_copy.groupby(group_cols, dropna=False).size().rename('total_messages')

    sent_counts = (
        df_copy.groupby(group_cols + ['sentiment'], dropna=False).size().unstack(fill_value=0)
    )

    avg_compound = df_copy.groupby(group_cols, dropna=False)['vader_compound'].mean().rename('avg_compound_score')

    summary = pd.concat([message_count, sent_counts, avg_compound], axis=1).fillna(0)

    # Calculate percentages
    summary['positive_pct'] = (summary.get('positive', 0) / summary['total_messages'] * 100).round(1)
    summary['neutral_pct'] = (summary.get('neutral', 0) / summary['total_messages'] * 100).round(1)
    summary['negative_pct'] = (summary.get('negative', 0) / summary['total_messages'] * 100).round(1)

    # Ensure all required columns exist and in order
    cols = [
        'total_messages',
        'positive', 'neutral', 'negative',
        'positive_pct', 'neutral_pct', 'negative_pct',
        'avg_compound_score'
    ]
    for c in cols:
        if c not in summary.columns:
            summary[c] = 0

    return summary[cols]
