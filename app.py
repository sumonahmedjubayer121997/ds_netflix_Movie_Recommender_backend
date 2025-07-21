from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load and process data
df = pd.read_csv("netflix_movies.csv")
df = df[['title', 'description', 'listed_in']].dropna()
df['combined'] = df['description'] + ' ' + df['listed_in']
df = df.reset_index(drop=True)

# Preprocess title index (lowercased and stripped)
df['title_clean'] = df['title'].str.lower().str.strip()
indices = pd.Series(df.index, index=df['title_clean'])

# Vectorize and compute cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title):
    title = title.lower().strip()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET"])
def index():
    return "✅ Netflix Movie Recommendation API is running!"

@app.route("/favicon.ico")
def favicon():
    return '', 204  # Silences the favicon.ico 404

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.json
    movie = data.get("title", "")
    results = recommend(movie)
    if results is None:
        return jsonify({"error": "Movie not found."}), 404
    return jsonify({"recommendations": results})

if __name__ == "__main__":
    app.run(debug=True)
