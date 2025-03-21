import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from .processing.main import AnimeRecommender
from .data_loader import DataLoader

# Initialize the data loader
data_loader = DataLoader()

# Load anime data
try:
    anime_data = data_loader.load_json_file('final_anime_data.json')
    anime_names = [anime['Name'] for anime in anime_data]
except Exception as e:
    print(f"Error loading data: {e}")
    anime_data = []
    anime_names = []

app = Flask(__name__)
CORS(app)

# Initialize the anime recommender with the already loaded data
recommender = AnimeRecommender(anime_data=anime_data)

@app.route("/")
def home():
    return render_template('base.html', title="Anime Recommender")

@app.route("/suggestions")
def get_suggestions():
    query = request.args.get("query", "").lower()
    if not query:
        return json.dumps([])
    
    # Filter anime names that start with the query
    suggestions = [name for name in anime_names if name.lower().startswith(query)]
    return json.dumps(suggestions)  # Return top 10 suggestions

@app.route("/recommendations")
def get_recommendations():
    anime_title = request.args.get("title")
    if not anime_title:
        return json.dumps([])
    
    recommendations = recommender.get_recommendations(anime_title)
    return json.dumps(recommendations)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5001)