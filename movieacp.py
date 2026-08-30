import random
from flask import Flask, jsonify, request
from textblob import TextBlob

app = Flask(__name__)

MOVIES = [
    {
        "title": "Inception",
        "genre": "Sci-Fi",
        "rating": 8.8,
        "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."
    },
    {
        "title": "The Shawshank Redemption",
        "genre": "Drama",
        "rating": 9.3,
        "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."
    },
    {
        "title": "The Dark Knight",
        "genre": "Action",
        "rating": 9.0,
        "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
    },
    {
        "title": "Paddington 2",
        "genre": "Comedy",
        "rating": 7.8,
        "overview": "Paddington, now happily settled with the Brown family, picks up a series of odd jobs to buy the perfect present for his aunt's 100th birthday, only for the gift to be stolen."
    },
    {
        "title": "Hereditary",
        "genre": "Horror",
        "rating": 7.3,
        "overview": "A grieving family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother, unraveling terrifying secrets."
    }
]

def analyze_sentiment(overview):
    polarity = TextBlob(overview).sentiment.polarity
    if polarity > 0.15:
        mood = "Uplifting"
    elif polarity < -0.05:
        mood = "Dark / Tense"
    else:
        mood = "Neutral"
    return {
        "polarity_score": round(polarity, 2),
        "detected_mood": mood
    }

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    genre = request.args.get('genre')
    min_rating = request.args.get('min_rating', type=float)
    mood = request.args.get('mood')
    random_flag = request.args.get('random', type=lambda v: v.lower() == 'true', default=False)

    filtered_movies = []
    for movie in MOVIES:
        sentiment = analyze_sentiment(movie['overview'])
        movie_data = {
            "title": movie["title"],
            "genre": movie["genre"],
            "rating": movie["rating"],
            "overview": movie["overview"],
            "sentiment_analysis": sentiment
        }
        filtered_movies.append(movie_data)

    if genre:
        filtered_movies = [m for m in filtered_movies if m['genre'].lower() == genre.lower()]

    if min_rating is not None:
        filtered_movies = [m for m in filtered_movies if m['rating'] >= min_rating]

    if mood:
        filtered_movies = [m for m in filtered_movies if m['sentiment_analysis']['detected_mood'].lower() == mood.lower()]

    if not filtered_movies:
        return jsonify({"message": "No movies found matching the criteria"}), 404

    if random_flag:
        return jsonify(random.choice(filtered_movies))

    return jsonify(filtered_movies)

if __name__ == '__main__':
    app.run(debug=True)