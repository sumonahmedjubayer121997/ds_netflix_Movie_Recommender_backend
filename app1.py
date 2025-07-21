import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("netflix_titles.csv")
df = df[['title', 'description', 'listed_in']].dropna()
df['combined_features'] = df['description'] + ' ' + df['listed_in']
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'].str.lower())

# Vectorize and compute similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommender function
def recommend(title):
    title = title.lower()
    if title not in indices:
        return ["Movie not found. Try another title."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Netflix Movie Recommender")
user_input = st.text_input("Enter a movie title:")

if user_input:
    recommendations = recommend(user_input)
    st.write("Top 5 Recommendations:")
    for movie in recommendations:
        st.write(f"👉 {movie}")
