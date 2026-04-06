import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\navod\Downloads\Netflix-Movie-Recommendation-System-main\Netflix-Movie-Recommendation-System-main\netflix_titles-1.csv", encoding="utf-8")

print("✅ Dataset loaded successfully!")
print("Total movies/shows:", df.shape[0])

df['description'] = df['description'].fillna("")
df['director'] = df['director'].fillna("Unknown")
df['country'] = df['country'].fillna("Unknown")
df['release_year'] = df['release_year'].fillna("Unknown")
df['language'] = df['country']

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['description'])
print("✅ TF-IDF matrix created! Shape:", tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
df = df.reset_index()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend_with_graph(title, cosine_sim=cosine_sim):
    if title not in indices:
        print("❌ Movie not found in dataset. Try another title.")
        return
    idx = indices[title]
    movie_info = df.iloc[idx]
    print(f"\n🎬 Movie Selected: {movie_info['title']}")
    print(f"📅 Release Year: {movie_info['release_year']}")
    print(f"🎥 Director: {movie_info['director']}")
    print(f"🌍 Country/Language: {movie_info['country']}")
    print(f"📝 Description: {movie_info['description']}\n")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = df['title'].iloc[movie_indices].tolist()
    scores = [score for _, score in sim_scores]

    print(f"📌 Top 5 Recommendations for '{title}':")
    for i, movie in enumerate(recommended_titles, start=1):
        print(f"{i}. {movie} (score: {scores[i-1]:.3f})")

    plt.figure(figsize=(8,4))
    sns.barplot(x=scores, y=recommended_titles)
    plt.xlabel("Similarity Score")
    plt.ylabel("Recommended Titles")
    plt.title(f"Top 5 Recommendations for '{title}'")
    plt.tight_layout()
    plt.show()

    peak_year = df['release_year'].value_counts().idxmax()
    peak_count = df['release_year'].value_counts().max()
    print(f"\n📊 The year with the MOST Netflix titles: {peak_year} ({peak_count} titles released)\n")

    plt.figure(figsize=(12,5))
    df['release_year'].value_counts().sort_index().plot(kind='line', marker="o")
    plt.axvline(x=peak_year, linestyle="--", label=f"Peak Year: {peak_year}")
    plt.title("Number of Netflix Titles Released per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

user_input = input("\n🎬 Enter a movie/show name: ")
recommend_with_graph(user_input)
