import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Set page and theme switch
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# 2. Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("imdb_top_1000.csv")
    df = df[['Series_Title', 'Overview', 'IMDB_Rating', 'Released_Year']].dropna()
    df.columns = ['title', 'overview', 'rating', 'year']
    return df

movies = load_data()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix)

# 3. Poster fetching function
def get_poster(title):
    api_key = "cc232f2c"  # Replace this
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    data = requests.get(url).json()
    return data.get("Poster", ""), data.get("imdbRating", "N/A")


# 4. Recommendation logic
def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = []
    for i, score in scores:
        title = movies.iloc[i]['title']
        rating = movies.iloc[i]['rating']
        year = movies.iloc[i]['year']
        poster, omdb_rating = get_poster(title)
        recommendations.append({
            "title": title,
            "poster": poster,
            "rating": rating,
            "year": year,
            "omdb_rating": omdb_rating
        })
    return recommendations

# 5. UI Elements
# st.markdown("## 🎥 Welcome to the IMDB Movie Recommender")
# st.markdown("Select a movie and get similar suggestions.")

# selected_movie = st.selectbox("🎞 Choose a movie:", movies['title'].sort_values().unique())

# if selected_movie:
#     st.markdown("### 🔍 Recommendations for you:")
#     recs = recommend(selected_movie)

#     cols = st.columns(5)
#     for col, rec in zip(cols, recs):
#         with col:
#             if rec["poster"]:
#                 st.image(rec["poster"], use_container_width=True)
#             st.markdown(f"{rec['title']}")
#             st.markdown(f"⭐ {rec['omdb_rating']} | 🎬 {rec['year']}")


# 5. UI Elements
st.markdown("## 🎥 Welcome to the IMDB Movie Recommender")
st.markdown("Select a movie and get similar suggestions.")

selected_movie = st.selectbox("🎞 Choose a movie:", movies['title'].sort_values().unique())

if selected_movie:
    st.markdown("### 🔍 Recommendations for you:")
    recs = recommend(selected_movie)

    cols = st.columns(5)
    for col, rec in zip(cols, recs):
        with col:
            if rec["poster"] and rec["poster"] != "N/A" and rec["poster"].startswith("http"):
                st.image(rec["poster"], use_container_width=True)
            else:
                st.markdown(
                    """
                    <div style='height:390px; display:flex; align-items:center; justify-content:center;
                                border:1px solid #ccc; border-radius:10px; background-color:#f2f2f2;'>
                        <span style='color:#888;'>Poster not found</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(f"{rec['title']}")
            st.markdown(f"⭐ {rec['omdb_rating']} | 🎬 {rec['year']}")





# Footer
st.markdown("---")
st.markdown("Made with ❤ by Sanjli Yadav (https://github.com/yourusername)", unsafe_allow_html=True)