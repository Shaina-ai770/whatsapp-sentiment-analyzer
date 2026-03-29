# UI Component Snippets - Ready to Use

This file contains ready-to-use code snippets for various UI components you can add to your dashboard.

---

## 📊 Chart Components

### 1. Animated Sentiment Gauge

```python
import plotly.graph_objects as go

def create_sentiment_gauge(sentiment_score):
    """
    Create an animated gauge showing overall sentiment
    sentiment_score: -1 to 1 (negative to positive)
    """
    # Convert to 0-100 scale
    gauge_value = (sentiment_score + 1) * 50

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Sentiment", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#fee2e2'},
                {'range': [33, 66], 'color': '#fef3c7'},
                {'range': [66, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2c3e50"}
    )

    return fig

# Usage
avg_sentiment = filtered_df['sentiment_score'].mean()
fig = create_sentiment_gauge(avg_sentiment)
st.plotly_chart(fig, use_container_width=True)
```

### 2. Comparison Bar Chart (Horizontal)

```python
def create_user_comparison_horizontal(df, metric='message_count'):
    """
    Create a horizontal bar chart comparing users
    """
    user_stats = df.groupby('author').size().sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=user_stats.values,
        y=user_stats.index,
        orientation='h',
        marker=dict(
            color=user_stats.values,
            colorscale='Viridis',
            line=dict(color='white', width=2)
        ),
        text=user_stats.values,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Messages: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Messages by User',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Number of Messages',
        yaxis_title='',
        height=max(300, len(user_stats) * 40),  # Dynamic height
        margin=dict(t=80, b=40, l=150, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.8)',
        showlegend=False
    )

    return fig

# Usage
st.plotly_chart(create_user_comparison_horizontal(filtered_df), use_container_width=True)
```

### 3. Stacked Area Chart

```python
def create_sentiment_area_chart(df, sentiment_col):
    """
    Create a stacked area chart for sentiment over time
    """
    df_copy = df.copy()
    df_copy['date'] = df_copy['datetime'].dt.date

    # Get sentiment counts by date
    sentiment_over_time = df_copy.groupby(['date', sentiment_col]).size().unstack(fill_value=0)

    fig = go.Figure()

    colors = {
        'positive': '#10b981',
        'negative': '#ef4444',
        'neutral': '#6b7280'
    }

    for sentiment in sentiment_over_time.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_over_time.index,
            y=sentiment_over_time[sentiment],
            name=sentiment.capitalize(),
            mode='lines',
            stackgroup='one',  # Enables stacking
            line=dict(width=0.5, color=colors.get(sentiment, '#6b7280')),
            fillcolor=colors.get(sentiment, '#6b7280'),
            hovertemplate='<b>%{x}</b><br>%{y} messages<extra></extra>'
        ))

    fig.update_layout(
        title='Sentiment Distribution Over Time (Stacked)',
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        hovermode='x unified',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.8)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    return fig

# Usage
st.plotly_chart(create_sentiment_area_chart(filtered_df, sentiment_col), use_container_width=True)
```

### 4. Radar Chart for Emotions

```python
def create_emotion_radar_chart(df, emotion_col, user=None):
    """
    Create a radar chart showing emotion distribution
    """
    if user:
        df = df[df['author'] == user]

    emotion_counts = df[emotion_col].value_counts()
    emotions = emotion_counts.index.tolist()
    values = emotion_counts.values.tolist()

    # Close the radar chart
    emotions_closed = emotions + [emotions[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=emotions_closed,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#667eea'),
        hovertemplate='<b>%{theta}</b><br>Count: %{r}<extra></extra>'
    ))

    title = f"Emotion Profile - {user}" if user else "Overall Emotion Profile"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            ),
            bgcolor='rgba(255,255,255,0.8)'
        ),
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    return fig

# Usage
if 'transformer_emotion' in filtered_df.columns:
    st.plotly_chart(create_emotion_radar_chart(filtered_df, 'transformer_emotion'), use_container_width=True)
```

### 5. Box Plot for Message Length Distribution

```python
def create_message_length_boxplot(df):
    """
    Create a box plot showing message length distribution by user
    """
    df_copy = df.copy()
    df_copy['message_length'] = df_copy['message'].str.len()

    fig = go.Figure()

    for user in df_copy['author'].unique():
        user_data = df_copy[df_copy['author'] == user]['message_length']
        fig.add_trace(go.Box(
            y=user_data,
            name=user,
            marker=dict(color=f'rgb({hash(user) % 256}, {(hash(user) * 2) % 256}, {(hash(user) * 3) % 256})'),
            boxmean='sd',  # Show mean and standard deviation
            hovertemplate='<b>%{y}</b> characters<extra></extra>'
        ))

    fig.update_layout(
        title='Message Length Distribution by User',
        yaxis_title='Message Length (characters)',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.8)',
        showlegend=True
    )

    return fig

# Usage
st.plotly_chart(create_message_length_boxplot(filtered_df), use_container_width=True)
```

---

## 🎨 Styled Components

### 1. Animated Metric Cards

```python
def metric_card(title, value, delta=None, icon="📊", color="#667eea"):
    """
    Create an animated metric card
    """
    delta_html = f"<div style='color: {color}; font-size: 0.9rem;'>{delta}</div>" if delta else ""

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, white 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: all 0.3s ease;
        text-align: center;
    '>
        <div style='font-size: 2.5rem;'>{icon}</div>
        <div style='color: #6b7280; font-size: 0.9rem; margin-top: 10px;'>{title}</div>
        <div style='font-size: 2rem; font-weight: 700; color: {color}; margin-top: 10px;'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Usage
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Total Messages", "1,234", "+10%", "📨")
with col2:
    metric_card("Active Users", "5", "All time", "👥")
with col3:
    metric_card("Positive Rate", "65%", "+5%", "😊")
with col4:
    metric_card("Avg Length", "42", "chars", "📏")
```

### 2. Progress Ring

```python
def progress_ring(percentage, label, size=120, color="#667eea"):
    """
    Create a circular progress indicator
    """
    st.markdown(f"""
    <div style='text-align: center;'>
        <svg width='{size}' height='{size}' viewBox='0 0 120 120'>
            <circle cx='60' cy='60' r='54' fill='none' stroke='#e0e0e0' stroke-width='8'/>
            <circle cx='60' cy='60' r='54' fill='none' stroke='{color}' stroke-width='8'
                    stroke-dasharray='{percentage * 3.39} 339'
                    stroke-linecap='round'
                    transform='rotate(-90 60 60)'
                    style='transition: stroke-dasharray 0.5s ease;'/>
            <text x='60' y='60' text-anchor='middle' dy='7'
                  style='font-size: 20px; font-weight: bold; fill: {color};'>
                {percentage}%
            </text>
        </svg>
        <div style='margin-top: 10px; color: #6b7280; font-weight: 600;'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

# Usage
col1, col2, col3 = st.columns(3)
with col1:
    progress_ring(75, "Positive", color="#10b981")
with col2:
    progress_ring(15, "Negative", color="#ef4444")
with col3:
    progress_ring(10, "Neutral", color="#6b7280")
```

### 3. Info Cards with Icons

```python
def info_card(title, content, icon="ℹ️", bg_color="#e0e7ff", border_color="#667eea"):
    """
    Create an informational card with icon
    """
    st.markdown(f"""
    <div style='
        background-color: {bg_color};
        border-left: 5px solid {border_color};
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
    '>
        <div style='display: flex; align-items: center;'>
            <div style='font-size: 2rem; margin-right: 15px;'>{icon}</div>
            <div>
                <div style='font-weight: 700; color: #2c3e50; font-size: 1.1rem;'>{title}</div>
                <div style='color: #4b5563; margin-top: 5px;'>{content}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Usage
info_card("Total Conversations", "Analyzed 1,234 messages across 5 users", "💬", "#dbeafe", "#3b82f6")
info_card("Time Period", "January 1, 2024 - December 31, 2024", "📅", "#fef3c7", "#f59e0b")
info_card("Sentiment Score", "Overall positive sentiment detected", "😊", "#d1fae5", "#10b981")
```

### 4. Comparison Cards

```python
def comparison_card(name, value, percentage, max_value, color="#667eea"):
    """
    Create a comparison card with progress bar
    """
    bar_width = (value / max_value * 100) if max_value > 0 else 0

    st.markdown(f"""
    <div style='
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    '>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <span style='font-weight: 600; color: #2c3e50;'>{name}</span>
            <span style='font-weight: 700; color: {color};'>{value}</span>
        </div>
        <div style='background: #f3f4f6; height: 8px; border-radius: 4px; overflow: hidden;'>
            <div style='
                background: {color};
                height: 100%;
                width: {bar_width}%;
                transition: width 0.5s ease;
            '></div>
        </div>
        <div style='text-align: right; margin-top: 5px; color: #6b7280; font-size: 0.85rem;'>
            {percentage}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# Usage
user_stats = filtered_df['author'].value_counts()
max_messages = user_stats.max()

st.markdown("### User Activity Ranking")
for user, count in user_stats.items():
    pct = (count / user_stats.sum() * 100)
    comparison_card(user, count, f"{pct:.1f}", max_messages)
```

---

## 🔍 Filter Components

### 1. Advanced Search Box

```python
def advanced_search():
    """
    Create an advanced search interface
    """
    with st.expander("🔎 Advanced Search", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input(
                "Search messages",
                placeholder="Enter keywords...",
                label_visibility="collapsed"
            )

        with col2:
            search_type = st.selectbox(
                "Type",
                ["Contains", "Exact", "Starts with", "Ends with"],
                label_visibility="collapsed"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            case_sensitive = st.checkbox("Case sensitive")
        with col2:
            regex_search = st.checkbox("Use regex")
        with col3:
            whole_word = st.checkbox("Whole word")

        return {
            'term': search_term,
            'type': search_type,
            'case_sensitive': case_sensitive,
            'regex': regex_search,
            'whole_word': whole_word
        }

# Usage
search_params = advanced_search()
if search_params['term']:
    # Apply search logic
    filtered_df = apply_search(df, search_params)
```

### 2. Date Range Presets

```python
def date_range_filter(df):
    """
    Create a date range filter with presets
    """
    st.markdown("#### 📅 Date Range")

    # Preset options
    preset = st.radio(
        "Quick select",
        ["All time", "Last 7 days", "Last 30 days", "Last 3 months", "Custom"],
        horizontal=True,
        label_visibility="collapsed"
    )

    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    if preset == "Last 7 days":
        start_date = max_date - pd.Timedelta(days=7)
        end_date = max_date
    elif preset == "Last 30 days":
        start_date = max_date - pd.Timedelta(days=30)
        end_date = max_date
    elif preset == "Last 3 months":
        start_date = max_date - pd.Timedelta(days=90)
        end_date = max_date
    elif preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
    else:  # All time
        start_date = min_date
        end_date = max_date

    return start_date, end_date

# Usage
start_date, end_date = date_range_filter(df)
filtered_df = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]
```

### 3. Multi-Criteria Filter

```python
def multi_criteria_filter(df):
    """
    Create a comprehensive filter panel
    """
    st.sidebar.markdown("## 🎯 Filters")

    filters = {}

    # User filter
    with st.sidebar.expander("👥 Users", expanded=True):
        all_users = df['author'].unique().tolist()
        filters['users'] = st.multiselect(
            "Select users",
            all_users,
            default=all_users
        )

    # Sentiment filter
    with st.sidebar.expander("😊 Sentiment", expanded=True):
        filters['sentiments'] = st.multiselect(
            "Select sentiments",
            ['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )

    # Message length filter
    with st.sidebar.expander("📏 Message Length", expanded=False):
        min_len, max_len = st.slider(
            "Character range",
            0, 500, (0, 500)
        )
        filters['min_length'] = min_len
        filters['max_length'] = max_len

    # Time of day filter
    with st.sidebar.expander("🕐 Time of Day", expanded=False):
        time_range = st.slider(
            "Hour range",
            0, 23, (0, 23)
        )
        filters['start_hour'] = time_range[0]
        filters['end_hour'] = time_range[1]

    # Apply filters button
    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun()

    return filters

# Usage
filters = multi_criteria_filter(df)
filtered_df = apply_filters(df, filters)
```

---

## 📱 Interactive Components

### 1. Sentiment Selector with Emojis

```python
def sentiment_selector():
    """
    Create an emoji-based sentiment selector
    """
    st.markdown("### Select Sentiment to Analyze")

    col1, col2, col3 = st.columns(3)

    sentiments = {
        'positive': {'emoji': '😊', 'color': '#10b981', 'label': 'Positive'},
        'neutral': {'emoji': '😐', 'color': '#6b7280', 'label': 'Neutral'},
        'negative': {'emoji': '😞', 'color': '#ef4444', 'label': 'Negative'}
    }

    selected = []

    with col1:
        if st.checkbox(sentiments['positive']['emoji'], value=True, key='pos'):
            selected.append('positive')
            st.markdown(f"<div style='text-align: center; color: {sentiments['positive']['color']}; font-weight: 600;'>{sentiments['positive']['label']}</div>", unsafe_allow_html=True)

    with col2:
        if st.checkbox(sentiments['neutral']['emoji'], value=True, key='neu'):
            selected.append('neutral')
            st.markdown(f"<div style='text-align: center; color: {sentiments['neutral']['color']}; font-weight: 600;'>{sentiments['neutral']['label']}</div>", unsafe_allow_html=True)

    with col3:
        if st.checkbox(sentiments['negative']['emoji'], value=True, key='neg'):
            selected.append('negative')
            st.markdown(f"<div style='text-align: center; color: {sentiments['negative']['color']}; font-weight: 600;'>{sentiments['negative']['label']}</div>", unsafe_allow_html=True)

    return selected

# Usage
selected_sentiments = sentiment_selector()
filtered_df = df[df['sentiment'].isin(selected_sentiments)]
```

### 2. Interactive Timeline

```python
def interactive_timeline(df):
    """
    Create an interactive timeline with range selection
    """
    df_copy = df.copy()
    df_copy['date'] = df_copy['datetime'].dt.date

    daily_counts = df_copy.groupby('date').size().reset_index(name='count')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_counts['date'],
        y=daily_counts['count'],
        mode='lines+markers',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6, color='#667eea'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>'
    ))

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#f3f4f6",
            activecolor="#667eea",
            x=0,
            y=1.1
        )
    )

    fig.update_layout(
        title='Message Timeline (Drag to zoom)',
        xaxis_title='Date',
        yaxis_title='Messages',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.8)',
        hovermode='x unified'
    )

    return fig

# Usage
st.plotly_chart(interactive_timeline(df), use_container_width=True)
```

### 3. User Comparison Selector

```python
def user_comparison_selector(df):
    """
    Create an interactive user comparison interface
    """
    st.markdown("### 👥 Compare Users")

    all_users = df['author'].unique().tolist()

    col1, col2 = st.columns(2)

    with col1:
        user1 = st.selectbox("User 1", all_users, key='user1')

    with col2:
        user2_options = [u for u in all_users if u != user1]
        user2 = st.selectbox("User 2", user2_options, key='user2') if user2_options else None

    if user1 and user2:
        # Create comparison
        user1_data = df[df['author'] == user1]
        user2_data = df[df['author'] == user2]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Messages",
                len(user1_data),
                delta=len(user1_data) - len(user2_data)
            )

        with col2:
            user1_pos = (user1_data['sentiment'] == 'positive').sum() / len(user1_data) * 100
            user2_pos = (user2_data['sentiment'] == 'positive').sum() / len(user2_data) * 100
            st.metric(
                f"{user1} Positivity",
                f"{user1_pos:.1f}%",
                delta=f"{user1_pos - user2_pos:.1f}%"
            )

        with col3:
            user1_avg_len = user1_data['message'].str.len().mean()
            user2_avg_len = user2_data['message'].str.len().mean()
            st.metric(
                "Avg Message Length",
                f"{user1_avg_len:.0f}",
                delta=f"{user1_avg_len - user2_avg_len:.0f}"
            )

        return user1, user2

    return None, None

# Usage
user1, user2 = user_comparison_selector(df)
```

---

## 🎯 Status Indicators

### 1. Sentiment Badge

```python
def sentiment_badge(sentiment, size="medium"):
    """
    Create a colored badge for sentiment
    """
    colors = {
        'positive': {'bg': '#d1fae5', 'text': '#065f46', 'emoji': '😊'},
        'negative': {'bg': '#fee2e2', 'text': '#991b1b', 'emoji': '😞'},
        'neutral': {'bg': '#f3f4f6', 'text': '#374151', 'emoji': '😐'}
    }

    sizes = {
        'small': {'padding': '4px 8px', 'font': '0.75rem'},
        'medium': {'padding': '8px 16px', 'font': '0.9rem'},
        'large': {'padding': '12px 24px', 'font': '1.1rem'}
    }

    color = colors.get(sentiment, colors['neutral'])
    size_style = sizes.get(size, sizes['medium'])

    return f"""
    <span style='
        background-color: {color['bg']};
        color: {color['text']};
        padding: {size_style['padding']};
        border-radius: 20px;
        font-weight: 600;
        font-size: {size_style['font']};
        display: inline-block;
    '>
        {color['emoji']} {sentiment.capitalize()}
    </span>
    """

# Usage
st.markdown(sentiment_badge('positive', 'medium'), unsafe_allow_html=True)
st.markdown(sentiment_badge('negative', 'small'), unsafe_allow_html=True)
st.markdown(sentiment_badge('neutral', 'large'), unsafe_allow_html=True)
```

### 2. Loading Animation

```python
def show_loading_animation(message="Processing..."):
    """
    Display a custom loading animation
    """
    st.markdown(f"""
    <div style='text-align: center; padding: 40px;'>
        <div style='
            border: 4px solid #f3f4f6;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px auto;
        '></div>
        <div style='color: #6b7280; font-weight: 600;'>{message}</div>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """, unsafe_allow_html=True)

# Usage
with st.spinner():
    show_loading_animation("Analyzing sentiment...")
    result = process_data()
```

---

## 💡 Tips for Implementation

1. **Copy and paste** the snippets directly into your code
2. **Adjust colors** to match your theme
3. **Combine components** to create unique layouts
4. **Test responsiveness** by resizing your browser
5. **Add error handling** for production use

---

## 🚀 Quick Start Examples

### Complete Dashboard Section

```python
# Complete sentiment overview section
st.markdown("## 📊 Sentiment Overview")

col1, col2 = st.columns([1, 2])

with col1:
    # Gauge
    avg_sentiment = filtered_df['sentiment_score'].mean()
    fig = create_sentiment_gauge(avg_sentiment)
    st.plotly_chart(fig, use_container_width=True)

    # Progress rings
    sentiment_dist = filtered_df['sentiment'].value_counts(normalize=True) * 100
    progress_ring(int(sentiment_dist.get('positive', 0)), "Positive", color="#10b981")

with col2:
    # Timeline
    fig = create_sentiment_timeline(filtered_df, 'sentiment')
    st.plotly_chart(fig, use_container_width=True)
```

### Complete Filter Sidebar

```python
with st.sidebar:
    st.markdown("# 🎯 Filters")

    # Date range
    start_date, end_date = date_range_filter(df)

    # Multi-criteria
    filters = multi_criteria_filter(df)

    # Advanced search
    search_params = advanced_search()

    # Apply button
    if st.button("Apply Filters", use_container_width=True):
        filtered_df = apply_all_filters(df, filters, search_params, start_date, end_date)
```

---

## 📚 Additional Resources

- All components are Plotly-based for maximum interactivity
- CSS can be customized in the `<style>` tags
- Components are mobile-responsive by default
- Use `st.columns()` for responsive layouts

Happy building!
