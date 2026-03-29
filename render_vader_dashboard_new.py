def render_vader_dashboard(df, sentiment_col):
    """Renders all UI components for the VADER analysis dashboard."""
    st.markdown("<h1 style='text-align: center;'>📊 VADER Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # Tab layout for VADER dashboard
    overview_tab, sentiment_tab, users_tab, words_tab, activity_tab, messages_tab = st.tabs(
        ["📈 Overview", "😊 Sentiment", "👥 Users", "☁️ Words", "📅 Activity", "💬 Messages"]
    )

    with overview_tab:
        st.subheader("Overall Sentiment and Activity")
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.plotly_chart(create_sentiment_pie_chart(df, sentiment_col), use_container_width=True)
        with col2:
            st.plotly_chart(create_message_heatmap(df), use_container_width=True)

    with sentiment_tab:
        st.subheader("Sentiment Trends")
        st.plotly_chart(create_sentiment_timeline(df, sentiment_col), use_container_width=True)
        st.markdown("---")
        
        st.subheader("Daily Average VADER Compound Score")
        st.plotly_chart(create_daily_average_compound_chart(df), use_container_width=True)
        st.markdown("---")

        st.subheader("Sentiment Distribution by User")
        st.plotly_chart(create_sentiment_comparison_chart(df, sentiment_col), use_container_width=True)

    with users_tab:
        st.subheader("User Analytics")
        st.markdown("#### 🔍 Individual User Deep Dive")
        
        # 1. Clean the author list properly
        user_list = (
            df["author"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        user_list.sort()

        selected_user_analysis = st.selectbox(
            "Select a user for detailed analysis",
            options=["Select User"] + user_list,
            key="vader_user_deep_dive_select"
        )

        # 2. Add safety check
        if selected_user_analysis == "Select User":
            st.info("Select a user to view analytics.")
        else:
            user_df = df[df['author'] == selected_user_analysis]
            
            if user_df.empty:
                st.warning(f"No messages from '{selected_user_analysis}' match the current filters.")
            else:
                col1, col2, col3 = st.columns(3, gap="small")
                with col1:
                    st.plotly_chart(create_user_activity_gauge(df, selected_user_analysis), use_container_width=True)
                with col2:
                    st.markdown(f"**{selected_user_analysis}'s Stats**")
                    st.metric("Total Messages", len(user_df))
                    st.metric("Avg Message Length", f"{user_df['message'].str.len().mean():.0f} chars")
                with col3:
                    st.markdown("**Sentiment Breakdown**")
                    sentiment_dist = user_df[sentiment_col].value_counts(normalize=True) * 100
                    for sent, pct in sentiment_dist.items():
                        color = SENTIMENT_COLORS.get(sent, '#6b7280')
                        st.markdown(f"<span style='color: {color};'>●</span> {sent.capitalize()}: {pct:.1f}%", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 User Comparison Table")
        from src.sentiment import per_user_summary
        user_stats = per_user_summary(df, by=['author'])
        st.dataframe(user_stats, use_container_width=True)

    with words_tab:
        st.subheader("Word Analysis")
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.markdown("#### Word Cloud")
            if 'message_raw' in df.columns:
                cleaned_messages = df.apply(clean_text_for_wordcloud, axis=1)
                text_for_wordcloud = ' '.join(cleaned_messages.tolist())
                
                if text_for_wordcloud.strip():
                    save_wordcloud(text_for_wordcloud, 'wc.png')
                    st.image('wc.png', use_container_width=True)
                else:
                    st.info("No meaningful words found to generate a word cloud.")
            else:
                st.warning("Raw message data not found for word cloud generation.")
        with col2:
            st.markdown("#### Top Words")
            st.plotly_chart(create_word_frequency_chart(df, top_n=15), use_container_width=True)
    
    with activity_tab:
        st.subheader("Activity Patterns")
        col1, col2 = st.columns(2, gap="small")
        with col1:
             if 'datetime' in df.columns and not df.empty:
                st.markdown("#### Messages by Day of Week")
                dow_counts = df['datetime'].dt.day_name().value_counts()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_counts = dow_counts.reindex(day_order)
                fig = px.bar(dow_counts, x=dow_counts.index, y=dow_counts.values, labels={'x': 'Day', 'y': 'Message Count'})
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'datetime' in df.columns and not df.empty:
                st.markdown("#### Messages by Hour of Day")
                hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
                fig = px.bar(hour_counts, x=[f"{h:02d}:00" for h in hour_counts.index], y=hour_counts.values, labels={'x': 'Hour', 'y': 'Message Count'})
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Average VADER Sentiment by Hour")
        st.plotly_chart(create_hourly_average_compound_chart(df), use_container_width=True)
        
    with messages_tab:
        st.subheader("Message Explorer")
        display_limit = st.number_input("Number of messages to display", 10, 1000, 50, 10)
        sort_by = st.selectbox("Sort by", ['datetime', sentiment_col, 'vader_compound'], index=0)
        sort_asc = st.checkbox("Sort Ascending", False)
        
        display_cols = ['datetime', 'author', 'message', sentiment_col, 'vader_compound']
        display_df = df[display_cols].copy()
        display_df = display_df.sort_values(by=sort_by, ascending=sort_asc).head(display_limit)
        
        st.dataframe(display_df, use_container_width=True)