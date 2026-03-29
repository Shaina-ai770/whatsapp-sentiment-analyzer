import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import matplotlib.dates as mdates



def plot_message_counts(df, output='message_counts.png'):
    counts = df.groupby(df['datetime'].dt.date).size()
    plt.figure(figsize=(10,4))
    counts.plot(kind='line')
    plt.title('Messages per day')
    plt.xlabel('Date')
    plt.ylabel('Messages')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_wordcloud(text, output='wordcloud.png'):
    """Generate and save a word cloud visualization from text data.

    This function creates a word cloud image based on the most frequently used words
    in the provided text. Word clouds are useful for quickly identifying popular terms
    and topics in the chat dataset. Words that appear more frequently are displayed
    in larger font sizes.

    Args:
        text (str): The text content to generate the word cloud from. Should be a
                   single string with all messages concatenated (words separated by spaces)
        output (str): File path where the word cloud image will be saved.
                     Default: 'wordcloud.png'

    Returns:
        None: The function saves the word cloud directly to the specified file path

    Image specifications:
        - Width: 800 pixels
        - Height: 400 pixels
        - Background color: white

    Example:
        >>> df = preprocess_df(parse_chat(chat_content))
        >>> all_text = ' '.join(df['message'].dropna())
        >>> save_wordcloud(all_text, 'my_wordcloud.png')

    Note:
        Ensure the output directory exists before calling this function, or
        provide a valid path with write permissions.
    """
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    wc.to_file(output)


def plot_message_counts_by_user(df, group_by='author', figsize=(8,4)):
    """Return a matplotlib Figure showing message counts per user.

    df: DataFrame with 'datetime' and group_by column
    """
    counts = df.groupby(group_by).size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    if counts.empty:
        # No data to plot: show a friendly message instead of failing
        ax.text(0.5, 0.5, 'No messages to display', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(group_by)
        ax.set_ylabel('Messages')
    ax.set_title('Messages per user')
    plt.tight_layout()
    return fig
