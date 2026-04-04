import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI   # DeepSeek uses OpenAI-compatible client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sensex Market Intelligence",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .headline-card {
        background: #f0f4ff;
        border-left: 3px solid #2166AC;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .source-label {
        font-size: 11px;
        color: #888;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = "sk-578e68468faa4ebd8d2fb61058403c4a"   # ← paste your key here
NEWS_PATH        = "india-news-headlines.csv"
EMBED_MODEL      = "all-MiniLM-L6-v2"
WINDOW_DAYS      = 7    # retrieve headlines ±7 days around query date

# ── Load everything once with caching ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data(show_spinner="Loading news headlines...")
def load_news():
    news = pd.read_csv(NEWS_PATH)
    # Handle both column name variants
    if 'publish_date' in news.columns:
        news['date'] = pd.to_datetime(news['publish_date'], format='%Y%m%d', errors='coerce')
    elif 'date' in news.columns:
        news['date'] = pd.to_datetime(news['date'], errors='coerce')
    if 'headline_text' in news.columns:
        news = news.rename(columns={'headline_text': 'headline'})
    news = news.dropna(subset=['date', 'headline'])
    news = news[news['date'] >= '2016-01-01']
    # Aggregate headlines per day
    daily = (news.groupby('date')['headline']
             .apply(lambda x: ' | '.join(x.astype(str).tolist()[:10]))
             .reset_index())
    return daily

@st.cache_data(show_spinner="Loading Sensex price data...")
def load_prices():
    raw = yf.download('^BSESN', start='2016-01-01',
                      end='2026-01-01', auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    df = raw[['Close']].reset_index()
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['return_pct'] = df['close'].pct_change() * 100
    return df

@st.cache_data(show_spinner="Building news vector index...")
def build_index(_embedder, news_df):
    headlines = news_df['headline'].tolist()
    embeddings = _embedder.encode(
        headlines, batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings

# ── RAG retrieval ──────────────────────────────────────────────────────────────
def retrieve_context(query, query_date, news_df, embeddings, embedder,
                     price_df, top_k=5):
    """
    Retrieve top-k relevant headlines near query_date,
    plus price statistics for that period.
    """
    # Filter news to ±WINDOW_DAYS around query date
    if query_date:
        start = pd.Timestamp(query_date) - timedelta(days=WINDOW_DAYS)
        end   = pd.Timestamp(query_date) + timedelta(days=WINDOW_DAYS)
        mask  = (news_df['date'] >= start) & (news_df['date'] <= end)
        local_news = news_df[mask].reset_index(drop=True)
        if len(local_news) == 0:
            # Fall back to global search if no local news
            local_news     = news_df
            local_embeddings = embeddings
        else:
            local_idx        = news_df[mask].index.tolist()
            local_embeddings = embeddings[local_idx]
    else:
        local_news       = news_df
        local_embeddings = embeddings

    # Semantic search
    query_emb = embedder.encode([query], convert_to_numpy=True)
    sims      = cosine_similarity(query_emb, local_embeddings)[0]
    top_idx   = np.argsort(sims)[::-1][:top_k]

    retrieved_headlines = []
    for idx in top_idx:
        row = local_news.iloc[idx]
        retrieved_headlines.append({
            'date'      : row['date'].strftime('%Y-%m-%d'),
            'headline'  : row['headline'][:300],
            'similarity': float(sims[idx])
        })

    # Price context
    price_context = ""
    if query_date:
        start_p = pd.Timestamp(query_date) - timedelta(days=30)
        end_p   = pd.Timestamp(query_date) + timedelta(days=7)
        window  = price_df[(price_df['date'] >= start_p) &
                           (price_df['date'] <= end_p)].copy()
        if len(window) > 1:
            p_start  = window['close'].iloc[0]
            p_end    = window['close'].iloc[-1]
            p_change = ((p_end - p_start) / p_start) * 100
            p_min    = window['close'].min()
            p_max    = window['close'].max()
            worst    = window['return_pct'].min()
            best     = window['return_pct'].max()
            price_context = (
                f"Price context for the period around {query_date}:\n"
                f"- Sensex moved from {p_start:,.0f} to {p_end:,.0f} "
                f"({p_change:+.1f}% over ~30 days)\n"
                f"- Range: {p_min:,.0f} – {p_max:,.0f}\n"
                f"- Worst single day: {worst:.2f}%\n"
                f"- Best single day: {best:.2f}%"
            )

    return retrieved_headlines, price_context


def build_prompt(query, headlines, price_context):
    headline_block = "\n".join([
        f"[{h['date']}] {h['headline']}"
        for h in headlines
    ])
    return f"""You are a financial analyst specialising in Indian equity markets.
You have access to real BSE Sensex price data and Indian news headlines.

{price_context}

Relevant news headlines:
{headline_block}

User question: {query}

Instructions:
- Answer using the price data and headlines as evidence
- Be specific — mention dates, percentage moves, and headline themes
- Keep the response concise (3-5 paragraphs)
- If the headlines don't fully explain the move, acknowledge that honestly
- End with one key takeaway
"""


def call_deepseek(prompt):
    client = OpenAI(
        api_key="sk-578e68468faa4ebd8d2fb61058403c4a",
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system",
             "content": "You are a financial analyst specialising in Indian equity markets (BSE Sensex). Provide clear, evidence-based analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3,   # low temp = more factual, less creative
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ── Load everything ────────────────────────────────────────────────────────────
embedder   = load_embedder()
news_df    = load_news()
price_df   = load_prices()
embeddings = build_index(embedder, news_df)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📈 Sensex Market Intelligence")
st.caption("RAG-powered chatbot — asks why the market moved, gets answers grounded in real news")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses **Retrieval-Augmented Generation (RAG)**:

    1. Your question is embedded using `all-MiniLM-L6-v2`
    2. Top-5 relevant headlines retrieved via cosine similarity
    3. Headlines + price data sent to **DeepSeek LLM**
    4. Response grounded in real evidence

    **Data sources**
    - BSE Sensex (Yahoo Finance, 2016–2026)
    - India News Headlines (Kaggle)

    **Models used**
    - Sentence-BERT for retrieval
    - DeepSeek-Chat for generation
    """)
    st.divider()
    st.subheader("Example questions")
    examples = [
        "Why did Sensex crash in March 2020?",
        "What happened to the market in May 2022?",
        "Why did Sensex rally in late 2023?",
        "What caused the volatility in October 2018?",
        "How did the 2024 elections affect the market?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state['prefill'] = ex

    st.divider()
    st.caption(f"News index: {len(news_df):,} daily records")
    st.caption(f"Price data: {len(price_df):,} trading days")

# Market overview strip
col1, col2, col3, col4 = st.columns(4)
latest     = price_df.iloc[-1]
prev       = price_df.iloc[-2]
day_change = ((latest['close'] - prev['close']) / prev['close']) * 100
yr_ago     = price_df[price_df['date'] <= latest['date'] - timedelta(days=365)].iloc[-1]
yr_change  = ((latest['close'] - yr_ago['close']) / yr_ago['close']) * 100

with col1:
    st.metric("Sensex (latest)", f"{latest['close']:,.0f}",
              f"{day_change:+.2f}% today")
with col2:
    st.metric("1-Year Return", f"{yr_change:+.1f}%")
with col3:
    st.metric("News records", f"{len(news_df):,}")
with col4:
    st.metric("Trading days", f"{len(price_df):,}")

st.divider()

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'prefill' not in st.session_state:
    st.session_state.prefill = ""

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant' and 'headlines' in msg:
            with st.expander("Retrieved headlines used as context"):
                for h in msg['headlines']:
                    st.markdown(f"""
                    <div class='headline-card'>
                        <div class='source-label'>{h['date']} — similarity: {h['similarity']:.3f}</div>
                        {h['headline'][:200]}
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
prefill = st.session_state.pop('prefill', "")
query   = st.chat_input("Ask about any Sensex market event...",
                        key="chat_input") or prefill

if query:
    # Show user message
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message("user"):
        st.markdown(query)

    # Detect date in query
    query_date = None
    date_hints = {
        'march 2020': '2020-03-23', 'covid': '2020-03-23',
        'may 2022': '2022-05-12',   'october 2018': '2018-10-26',
        '2023': '2023-12-01',       '2024 election': '2024-06-04',
        'june 2024': '2024-06-04',  'january 2024': '2024-01-15',
        'budget 2023': '2023-02-01','demonetization': '2016-11-08',
    }
    query_lower = query.lower()
    for hint, date in date_hints.items():
        if hint in query_lower:
            query_date = date
            break

    # Retrieve context
    with st.spinner("Searching news index..."):
        headlines, price_context = retrieve_context(
            query, query_date, news_df,
            embeddings, embedder, price_df
        )

    # Build prompt and stream response
    prompt = build_prompt(query, headlines, price_context)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response        = ""

        try:
            for chunk in call_deepseek(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # Show retrieved headlines
            with st.expander("Retrieved headlines used as context"):
                for h in headlines:
                    st.markdown(f"""
                    <div class='headline-card'>
                        <div class='source-label'>{h['date']} — similarity: {h['similarity']:.3f}</div>
                        {h['headline'][:200]}
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            full_response = f"API error: {str(e)}\n\nCheck your DeepSeek API key."
            response_placeholder.error(full_response)

    # Save to history
    st.session_state.messages.append({
        'role'     : 'assistant',
        'content'  : full_response,
        'headlines': headlines
    })