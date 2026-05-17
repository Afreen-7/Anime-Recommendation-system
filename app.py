import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───
st.set_page_config(
    page_title="AnimeFlux — Discover Your Next Obsession",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

/* ── Global Overrides ── */
.stApp {
    background: linear-gradient(180deg, #0D0B18 0%, #1a1625 40%, #0D0B18 100%);
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1rem !important;
    max-width: 1200px;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1rem;
    position: relative;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -60px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 400px;
    background: radial-gradient(ellipse, rgba(255,107,138,0.15) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.brand-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #FF6B8A 0%, #FF8E53 50%, #FF6B8A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 40px rgba(255,107,138,0.3);
}

.hero-headline {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 3px;
    margin-bottom: 0.3rem;
    position: relative;
    z-index: 1;
}

.hero-headline .accent {
    background: linear-gradient(90deg, #FF6B8A, #FF8E53);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 0.95rem;
    color: #8B8697;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
}

/* ── Search Area ── */
.search-wrapper {
    max-width: 600px;
    margin: 0 auto 2rem;
    position: relative;
    z-index: 1;
}

/* Style the selectbox */
div[data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,107,138,0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.3s ease !important;
}

div[data-baseweb="select"]:hover,
div[data-baseweb="select"]:focus-within {
    border-color: #FF6B8A !important;
    box-shadow: 0 0 20px rgba(255,107,138,0.2) !important;
}

div[data-baseweb="select"] > div {
    background: transparent !important;
    color: #fff !important;
}

/* Style the button */
.stButton > button {
    background: linear-gradient(135deg, #FF6B8A 0%, #FF8E53 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 2.5rem !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(255,107,138,0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,107,138,0.5) !important;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 4px;
    color: #FF6B8A;
    text-align: center;
    margin: 2rem 0 1rem;
    text-transform: uppercase;
}

.section-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #FF6B8A, transparent);
    margin: 0 auto 1.5rem;
}

/* ── Genre Pills ── */
.genre-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin: 1rem auto 2rem;
    max-width: 600px;
}

.genre-pill {
    background: rgba(255,107,138,0.08);
    border: 1px solid rgba(255,107,138,0.25);
    border-radius: 25px;
    padding: 8px 22px;
    color: #e8dff5;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    cursor: pointer;
    backdrop-filter: blur(10px);
}

.genre-pill:hover {
    background: rgba(255,107,138,0.2);
    border-color: #FF6B8A;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255,107,138,0.2);
}

/* ── Anime Cards ── */
.anime-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(20px);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}

.anime-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FF6B8A, #FF8E53, #FF6B8A);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.anime-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255,107,138,0.3);
    box-shadow: 0 15px 40px rgba(0,0,0,0.4), 0 0 30px rgba(255,107,138,0.1);
}

.anime-card:hover::before {
    opacity: 1;
}

.anime-card-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    display: block;
}

.anime-card-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}

.anime-card-genre {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 0.8rem;
}

.genre-tag {
    background: rgba(255,107,138,0.12);
    border: 1px solid rgba(255,107,138,0.2);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.7rem;
    color: #FF8E9E;
    font-weight: 500;
    letter-spacing: 0.3px;
}

.anime-card-meta {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.78rem;
    color: #8B8697;
}

.meta-item .meta-icon {
    font-size: 0.9rem;
}

.meta-value {
    color: #c4bdd4;
    font-weight: 600;
}

.rating-badge {
    background: linear-gradient(135deg, #FF6B8A, #FF8E53);
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #fff;
}

/* ── Spotlight Card ── */
.spotlight-card {
    background: linear-gradient(145deg, rgba(255,107,138,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,107,138,0.2);
    border-radius: 24px;
    padding: 2.5rem;
    margin: 1rem 0 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.spotlight-card::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,107,138,0.1) 0%, transparent 70%);
    pointer-events: none;
}

.spotlight-rank {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #FF6B8A;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}

.spotlight-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 0.8rem;
    line-height: 1.2;
}

.spotlight-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 1rem;
}

.spotlight-tag {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 6px 16px;
    font-size: 0.8rem;
    color: #c4bdd4;
    font-weight: 500;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 3rem 1rem 2rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

.footer-brand {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #FF6B8A, #FF8E53);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.footer-text {
    font-size: 0.8rem;
    color: #5a5466;
    letter-spacing: 1px;
}

/* ── No Results ── */
.no-results {
    text-align: center;
    padding: 3rem;
    color: #5a5466;
    font-size: 1.1rem;
}

.no-results .emoji {
    font-size: 3rem;
    display: block;
    margin-bottom: 1rem;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin: 2rem 0;
    padding: 1.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

.stat-item {
    text-align: center;
}

.stat-number {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF6B8A, #FF8E53);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 0.75rem;
    color: #5a5466;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #0D0B18;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,107,138,0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255,107,138,0.5);
}

/* Hide label for selectbox */
.stSelectbox label {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Loading & ML Engine ───
@st.cache_data
def load_data():
    anime = pd.read_csv("anime.csv")
    anime = anime[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']].dropna(subset=['name', 'genre'])
    anime.drop_duplicates(subset='name', inplace=True)
    anime.reset_index(drop=True, inplace=True)
    return anime


@st.cache_resource
def build_similarity(_anime_df):
    genre_clean = _anime_df['genre'].fillna('').str.replace(' ', '')
    cv = CountVectorizer(max_features=3000, stop_words='english')
    vectors = cv.fit_transform(genre_clean)
    sim = cosine_similarity(vectors)
    return sim


def recommend(anime_name, anime_df, similarity, top_n=5):
    """Get top N similar anime by genre cosine similarity."""
    matches = anime_df[anime_df['name'] == anime_name]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    distances = similarity[idx]
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n + 1]
    indices = [i[0] for i in anime_list]
    return anime_df.iloc[indices]


def get_by_genre(genre_keyword, anime_df, top_n=6):
    """Filter anime by genre keyword, sorted by rating."""
    filtered = anime_df[anime_df['genre'].str.contains(genre_keyword, case=False, na=False)]
    filtered = filtered.sort_values('rating', ascending=False).head(top_n)
    return filtered


# ─── Emoji icons for anime types ───
TYPE_ICONS = {
    'TV': '📺', 'Movie': '🎬', 'OVA': '💿', 'Special': '✨',
    'ONA': '🌐', 'Music': '🎵',
}

GENRE_ICONS = {
    'Action': '⚡', 'Comedy': '😂', 'Drama': '🎭', 'Fantasy': '🧙',
    'Romance': '💖', 'Sci-Fi': '🚀', 'Horror': '👻', 'Mystery': '🔍',
    'Adventure': '🗺️', 'Slice of Life': '🌸', 'Magic': '✨',
    'Supernatural': '👁️', 'Isekai': '🌀', 'Sports': '⚽',
    'Music': '🎵', 'Mecha': '🤖', 'Military': '🎖️', 'School': '🏫',
    'Shounen': '💪', 'Shoujo': '🌹',
}


def get_anime_icon(genres_str):
    """Pick a relevant emoji based on the first matching genre."""
    if pd.isna(genres_str):
        return '🎌'
    for genre, icon in GENRE_ICONS.items():
        if genre.lower() in genres_str.lower():
            return icon
    return '🎌'


def render_anime_card(row):
    """Render a single anime card as HTML."""
    icon = get_anime_icon(row.get('genre', ''))
    name = row.get('name', 'Unknown')
    genres = row.get('genre', '')
    anime_type = row.get('type', 'N/A')
    episodes = row.get('episodes', 'N/A')
    rating = row.get('rating', 0)
    members = row.get('members', 0)
    type_icon = TYPE_ICONS.get(anime_type, '📺')

    # Build genre tags HTML
    genre_list = [g.strip() for g in str(genres).split(',')][:4]
    genre_tags = ''.join([f'<span class="genre-tag">{g}</span>' for g in genre_list])

    # Format members
    if isinstance(members, (int, float)) and not pd.isna(members):
        if members >= 1_000_000:
            members_str = f"{members/1_000_000:.1f}M"
        elif members >= 1_000:
            members_str = f"{members/1_000:.0f}K"
        else:
            members_str = str(int(members))
    else:
        members_str = "N/A"

    rating_display = f"{float(rating):.2f}" if not pd.isna(rating) else "N/A"

    return f"""
    <div class="anime-card">
        <span class="anime-card-icon">{icon}</span>
        <div class="anime-card-title">{name}</div>
        <div class="anime-card-genre">{genre_tags}</div>
        <div class="anime-card-meta">
            <span class="rating-badge">⭐ {rating_display}</span>
            <span class="meta-item">
                <span class="meta-icon">{type_icon}</span>
                <span class="meta-value">{anime_type}</span>
            </span>
            <span class="meta-item">
                <span class="meta-icon">📀</span>
                <span class="meta-value">{episodes} eps</span>
            </span>
            <span class="meta-item">
                <span class="meta-icon">👥</span>
                <span class="meta-value">{members_str}</span>
            </span>
        </div>
    </div>
    """


# ─── Load Data ───
anime_df = load_data()
similarity = build_similarity(anime_df)

# ─── HERO SECTION ───
st.markdown("""
<div class="hero-container">
    <div class="brand-title">ANIMEFLUX</div>
    <div class="hero-headline">DISCOVER YOUR <span class="accent">NEXT</span></div>
    <div class="hero-headline"><span class="accent">OBSESSION</span></div>
    <p class="hero-sub">AI-powered anime recommendations based on genre similarity</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats Bar ───
total_anime = len(anime_df)
total_genres = len(set(g.strip() for genres in anime_df['genre'].dropna() for g in genres.split(',')))
avg_rating = anime_df['rating'].mean()

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-number">{total_anime:,}</div>
        <div class="stat-label">Anime</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">{total_genres}</div>
        <div class="stat-label">Genres</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">{avg_rating:.1f}</div>
        <div class="stat-label">Avg Rating</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Search ───
st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)
anime_names = anime_df['name'].sort_values().tolist()
selected_anime = st.selectbox(
    "Search anime",
    options=[""] + anime_names,
    index=0,
    placeholder="🔍  Search for an anime...",
    label_visibility="collapsed",
)
get_rec = st.button("🚀  GET RECOMMENDATIONS", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─── Quick Start Genre Pills ───
st.markdown('<div class="section-header">Quick Start</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

genre_cols = st.columns(5)
genre_buttons = {
    0: ("⚡ Action", "Action"),
    1: ("🌀 Isekai", "Isekai"),
    2: ("🌸 Slice of Life", "Slice of Life"),
    3: ("✨ Magic", "Magic"),
    4: ("😂 Comedy", "Comedy"),
}

selected_genre = None
for i, col in enumerate(genre_cols):
    label, genre_key = genre_buttons[i]
    with col:
        if st.button(label, key=f"genre_{genre_key}", use_container_width=True):
            selected_genre = genre_key

# ─── Recommendation Results ───
if get_rec and selected_anime:
    st.markdown(f'<div class="section-header">Recommendations for "{selected_anime}"</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    results = recommend(selected_anime, anime_df, similarity, top_n=6)

    if results.empty:
        st.markdown("""
        <div class="no-results">
            <span class="emoji">😔</span>
            No recommendations found. Try a different anime!
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                st.markdown(render_anime_card(row), unsafe_allow_html=True)

elif selected_genre:
    st.markdown(f'<div class="section-header">{selected_genre} Anime</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    genre_results = get_by_genre(selected_genre, anime_df, top_n=6)

    if genre_results.empty:
        st.markdown("""
        <div class="no-results">
            <span class="emoji">🔍</span>
            No anime found for this genre.
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(genre_results.iterrows()):
            with cols[idx % 3]:
                st.markdown(render_anime_card(row), unsafe_allow_html=True)

# ─── Trending Now (default view) ───
if not (get_rec and selected_anime) and not selected_genre:
    st.markdown('<div class="section-header">Trending Now</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    trending = anime_df.sort_values('rating', ascending=False).head(6)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(trending.iterrows()):
        with cols[idx % 3]:
            st.markdown(render_anime_card(row), unsafe_allow_html=True)

# ─── Seasonal Spotlight ───
st.markdown('<div class="section-header">Seasonal Spotlight</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

top_anime = anime_df.sort_values('rating', ascending=False).iloc[0]
spotlight_genres = [g.strip() for g in str(top_anime['genre']).split(',')]
spotlight_tags = ''.join([f'<span class="spotlight-tag">{g}</span>' for g in spotlight_genres])
spotlight_rating = f"{float(top_anime['rating']):.2f}" if not pd.isna(top_anime['rating']) else "N/A"
members_count = top_anime['members']
if isinstance(members_count, (int, float)) and not pd.isna(members_count):
    if members_count >= 1_000_000:
        members_display = f"{members_count/1_000_000:.1f}M members"
    elif members_count >= 1_000:
        members_display = f"{members_count/1_000:.0f}K members"
    else:
        members_display = f"{int(members_count)} members"
else:
    members_display = ""

st.markdown(f"""
<div class="spotlight-card">
    <div class="spotlight-rank">🏆 #1 Highest Rated</div>
    <div class="spotlight-title">{top_anime['name']}</div>
    <div class="anime-card-meta" style="border:none;padding:0;margin:0.5rem 0;">
        <span class="rating-badge">⭐ {spotlight_rating}</span>
        <span class="meta-item">
            <span class="meta-icon">{TYPE_ICONS.get(top_anime.get('type',''), '📺')}</span>
            <span class="meta-value">{top_anime.get('type', 'N/A')}</span>
        </span>
        <span class="meta-item">
            <span class="meta-icon">📀</span>
            <span class="meta-value">{top_anime.get('episodes', 'N/A')} eps</span>
        </span>
        <span class="meta-item">
            <span class="meta-icon">👥</span>
            <span class="meta-value">{members_display}</span>
        </span>
    </div>
    <div class="spotlight-meta">{spotlight_tags}</div>
</div>
""", unsafe_allow_html=True)

# ─── Footer ───
st.markdown("""
<div class="footer">
    <div class="footer-brand">ANIMEFLUX</div>
    <div class="footer-text">Powered by Content-Based Filtering & Cosine Similarity</div>
    <div class="footer-text" style="margin-top: 0.5rem;">© 2026 AnimeFlux. Built with ❤️ and Streamlit</div>
</div>
""", unsafe_allow_html=True)
