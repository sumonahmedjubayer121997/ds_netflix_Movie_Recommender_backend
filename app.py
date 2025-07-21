from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ✅ TEMPORARY: Allow ALL origins for debug (switch to Vercel origin later)
CORS(app)

# ─────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv("netflix_movies.csv")
    except Exception as e:
        raise FileNotFoundError("❌ CSV file not found or unreadable.")

    df = df[['title', 'description', 'listed_in']].dropna()
    df['combined'] = df['description'] + ' ' + df['listed_in']
    df = df.reset_index(drop=True)
    df['title_clean'] = df['title'].str.lower().str.strip()
    return df

def build_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title_clean'])
    return cosine_sim, indices

def recommend(title):
    df = load_data()
    cosine_sim, indices = build_model(df)
    title = title.lower().strip()

    if title not in indices:
        return None

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# ─────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return "✅ Netflix Movie Recommendation API is running!"

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    try:
        data = request.json
        movie = data.get("title", "")
        results = recommend(movie)

        if results is None:
            return jsonify({"error": "Movie not found."}), 404

        return jsonify({"recommendations": results})

    except Exception as e:
        print("🔥 Internal Server Error:", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ─────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
