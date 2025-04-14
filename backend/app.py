import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from processing.main import AnimeRecommender

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load anime names for autocomplete
with open(os.path.join(current_directory, 'data/final_anime_data.json'), 'r') as f:
    anime_data = json.load(f)
    anime_names = [anime['Name'] for anime in anime_data]

app = Flask(__name__)
CORS(app)

# Initialize the anime recommender
recommender = AnimeRecommender()

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