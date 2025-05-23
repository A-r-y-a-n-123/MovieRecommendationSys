import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load dataset
movies = pd.read_csv('../data/tmdb_5000_movies.csv')
movies = movies[['title', 'overview']].fillna('')

# Clean and vectorize
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip()

movies['overview'] = movies['overview'].apply(clean_text)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Normalize movie titles for comparison
movies['title_lower'] = movies['title'].str.lower()

# Recommend function
def recommend_movies(title, top_n=10):
    title = title.lower()
    
    if title not in movies['title_lower'].values:
        return "Movie not found."
    
    idx = movies[movies['title_lower'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended = [movies['title'].iloc[i[0]] for i in scores]
    return recommended

# test
if __name__ == '__main__':
    user_input = input("Enter name of the movie: ")
    print("\nRecommended Movies:")
    print('\n'.join(recommend_movies(user_input)))
