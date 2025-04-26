import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from processing.main import AnimeRecommender
from processing.adhoc import AdHoc

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

@app.route('/filtered_recommendations', methods=['POST'])
def filtered_recommendations():
    try:
        data = request.get_json()
        genres = data.get('genres', [])
        episodes = data.get('episodes', [])
        studios = data.get('studios', [])
        ratings = data.get('ratings', [])
        description = data.get('description', '')

        # Initialize AdHoc with anime data
        adhoc = AdHoc(anime_data)
        
        # Get recommendations
        recommendations = adhoc.get_recommendations(
            genres=genres,
            episodes=episodes,
            studios=studios,
            ratings=ratings,
            description=description,
            
        )
        
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in filtered_recommendations: {str(e)}")
        return jsonify([]), 500

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5001)