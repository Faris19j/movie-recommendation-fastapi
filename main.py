# Import necessary libraries
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🚀 Load and process ratings data
ratings = pd.read_csv('u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
print("Ratings data preview:\n", ratings.head())

# 🎬 Load movie data with genres
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', names=[
    'movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
])
print("Movies data preview:\n", movies[['movieId', 'title']].head())

# ✅ Combine title and genres for better recommendations
genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies['genres'] = movies[genre_columns].apply(lambda x: ' '.join([genre for genre, val in zip(genre_columns, x) if val == 1]), axis=1)
movies['combined_features'] = movies['title'] + " " + movies['genres']

# 🔍 Build recommendation engine
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🎯 Recommendation function
def recommend_movie(title, cosine_sim=cosine_sim):
    # Handle case-insensitive movie search
    title = title.lower()
    movies['lower_title'] = movies['title'].str.lower()

    if title not in movies['lower_title'].values:
        return ["❌ Movie not found. Please check the spelling."]

    idx = movies[movies['lower_title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices].to_dict(orient='records')

# 🚀 FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": None})

@app.post("/", response_class=HTMLResponse)
async def get_recommendations(request: Request, movie: str = Form(...)):
    recommendations = recommend_movie(movie)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "recommendations": recommendations,
        "reason": f"✨ Because you liked **{movie}**, which shares similar genres or themes!"
    })
