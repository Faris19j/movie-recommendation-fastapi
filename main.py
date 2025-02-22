# Import necessary libraries
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and convert the ratings data
ratings = pd.read_csv('u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
ratings.to_csv('ratings.csv', index=False)
print("Ratings data preview:\n", ratings.head())

# Load and convert the movies data
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', names=[
    'movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
])

# We only need movieId and title for this project
movies[['movieId', 'title']].to_csv('movies.csv', index=False)
print("Movies data preview:\n", movies[['movieId', 'title']].head())

# Load the cleaned CSV files
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess dataset
movies['combined_features'] = movies['title']

# Build recommendation engine
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(title, cosine_sim=cosine_sim):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# FastAPI setup
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
        "reason": f"Because you liked {movie}, which shares similar genres or themes."
    })
