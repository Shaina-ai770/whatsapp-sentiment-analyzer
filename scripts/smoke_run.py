
from src.parser import parse_chat
from src.preprocess import preprocess_df
from src.sentiment import apply_vader

# Small inline chat sample to exercise the pipeline
sample = """12/31/20, 9:12 PM - Alice: Happy New Year! 🎉\n12/31/20, 9:13 PM - Bob: Thanks, you too! 🙂\n12/31/20, 9:14 PM - Carol: I'm worried about the deadline :(\n"""

p = 'scripts/_sample_chat.txt'
with open(p, 'w', encoding='utf-8') as f:
    f.write(sample)

print('Parsing...')
df = parse_chat(p)
print(df[['datetime','author','message']])

print('\nPreprocessing...')
df = preprocess_df(df)
print(df[['author','message','emojis','word_count']])

print('\nSentiment...')
df = apply_vader(df)
print(df[['author','message','vader_compound','sentiment']])

print('\nSummary:')
print(df['sentiment'].value_counts())

print('\nPer-user summary:')
from src.sentiment import per_user_summary
print(per_user_summary(df, by=['author','mobile']))
