import pandas as pd
from textblob import TextBlob

# Load dataset
df = pd.read_csv("imdb_top_1000.csv")

# Get unique genres
genres = sorted(set(
    g.strip()
    for genre_list in df["Genre"].dropna()
    for g in genre_list.split(", ")
))

# Recommendation function
def recommend(genre, rating):
    movies = df[df["Genre"].str.contains(genre, case=False, na=False, regex=False)]
    if rating:
        movies = movies[movies["IMDB_Rating"] >= rating]

    recommendations = []
    for _, movie in movies.iterrows():
        overview = movie["Overview"]
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity
        recommendations.append((movie["Series_Title"], polarity))
        if len(recommendations) == 5:
            break
    return recommendations

# Main program
print("Movie Recommendation System")
name = input("Enter your name: ")

print("\nAvailable Genres:")
for i, genre in enumerate(genres, 1):
    print(i, genre)

choice = int(input("\nChoose a genre number: "))
genre = genres[choice - 1]

input("How are you feeling today? ")
rating = float(input("Enter minimum IMDb rating: "))

recommendations = recommend(genre, rating)

print(f"\nMovie recommendations for {name}:")
for i, (movie, polarity) in enumerate(recommendations, 1):
    print(f"{i}. {movie} (Polarity: {polarity:.2f})")

