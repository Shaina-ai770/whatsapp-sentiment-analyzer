import sys

file_path = 'streamlit_app_enhanced.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

# 1. Identify and remove the massive duplication
indices = [i for i, line in enumerate(lines) if line.strip() == 'import streamlit as st']
duplication_start_index = -1
if len(indices) >= 2:
    # Use the second occurrence as the duplication start
    # But wait, the second occurrence might be in the middle of some other logic?
    # No, our previous read_file showed it at ~1936.
    for idx in indices:
        if 1930 <= idx <= 1950:
            duplication_start_index = idx
            break

if duplication_start_index == -1:
    print("Could not find duplication_start_index")
    sys.exit(1)

# Part 1: Start of file up to the duplication point
part1 = lines[:duplication_start_index]

# Part 2: The block that was supposed to be there instead of the duplication
part2 = [
    '                html_content += "</table></body></html>"\n',
    '                components.html(html_content, height=450, scrolling=True)\n',
    '            else:\n',
    '                st.warning("No URLs or link-sharing emojis (🔗, 🌐, 📎) found in the current selection.")\n',
    '                st.info(f"Analyzed {len(filtered_df)} messages. Try adjusting your filters in the sidebar.")\n',
    '\n'
]

# Part 3: The rest of the file after the duplication
# Search for '        with sentiment_tab:' after duplication
search_start = duplication_start_index + 2000 
sentiment_tab_index = -1
for i in range(search_start, len(lines)):
    if 'with sentiment_tab:' in lines[i] and 'with overview_tab,' not in lines[i]:
        sentiment_tab_index = i
        break

if sentiment_tab_index != -1:
    part3 = lines[sentiment_tab_index:]
else:
    print("Could not find sentiment_tab_index")
    sys.exit(1)

new_lines = part1 + part2 + part3
content = "".join(new_lines)

# Fix 2: process_data missing 'if _transformer_language != 'English':'
old_vader_block = """            # Determine models and strategy based on language
                start_time_vader = time.time()"""
new_vader_block = """            # Determine models and strategy based on language
            if _transformer_language != 'English':
                start_time_vader = time.time()"""
content = content.replace(old_vader_block, new_vader_block)

# Fix 4: model_performance_tab missing 'if transformer_language != 'English':'
old_perf_block = """                            transformer_language = st.session_state.get('transformer_language_selector', 'English')
                            s_model = None
                            e_model = None
                                s_model = "nlptown/bert-base-multilingual-uncased-sentiment"
                                e_model = "nlptown/bert-base-multilingual-uncased-sentiment"
                            else:"""
new_perf_block = """                            transformer_language = st.session_state.get('transformer_language_selector', 'English')
                            s_model = None
                            e_model = None
                            if transformer_language != 'English':
                                s_model = "nlptown/bert-base-multilingual-uncased-sentiment"
                                e_model = "nlptown/bert-base-multilingual-uncased-sentiment"
                            else:"""
content = content.replace(old_perf_block, new_perf_block)

with open(file_path, 'w') as f:
    f.write(content)

print("Repair completed successfully.")
