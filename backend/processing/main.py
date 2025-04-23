"""
Main module for the Anime Recommendation System.
This module implements a hybrid recommendation system that combines multiple approaches:
1. Genre-based similarity using Jaccard similarity
2. Reviewer sentiment analysis
3. SVD-based content similarity

The system uses a pipeline approach where each step refines the recommendations
from the previous step to provide more accurate and relevant suggestions.
"""

import os
import json
from typing import List, Dict
from genre_jaccard_sim import GenreJaccardSim
from reviewer_sentiment import ReviewerSentiment
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from svd import Svd

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
backend_directory = os.path.dirname(current_directory)  # Only go up one level to get to backend

# Load the main anime dataset containing all anime information
# This includes details like title, genres, ratings, etc.
with open(os.path.join(backend_directory, 'data/final_anime_data.json'), 'r') as f:
    anime_data = json.load(f)

# Load the mapping of anime titles to their indices in the dataset
# This is used for quick lookups when processing recommendations
with open(os.path.join(backend_directory, 'data/anime_to_index.json'), 'r') as f:
    anime_to_index = json.load(f)

# Load the reviewer-anime relationship data, which contains a map of anime id to
# a list of id of reviewers who gave it a score of 9+
# These mappings help in finding similar anime based on reviewer preferences
with open(os.path.join(backend_directory, 'data/anime_to_reviewer.json'), 'r') as f:
    anime_to_reviewer = json.load(f)
        
with open(os.path.join(backend_directory, 'data/reviewer_to_anime.json'), 'r') as f:
    reviewer_to_anime = json.load(f)

# Load the mapping of indices to anime IDs
# This helps in maintaining consistency between different data representations
with open(os.path.join(backend_directory, 'data/index_to_anime_id.json'), 'r') as f:
    index_to_anime_id = json.load(f)

class AnimeRecommender:
    """
    Main class for the anime recommendation system.
    Implements a hybrid recommendation approach combining multiple similarity metrics.
    """
    
    def __init__(self, anime_data_path='../data/final_anime_data.json'):
        # Get absolute path to data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(current_dir, anime_data_path))
        
        # Verify file exists before loading
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Anime data file not found at: {data_path}")
            
        print(f"Loading anime data from: {data_path}")  # Debug confirmation
        
        try:
            with open(data_path, 'r') as f:
                self.anime_data = json.load(f)
            with open('../data/anime_to_index.json', 'r') as r:
                self.anime_to_index = json.load(r)
            print(f"Successfully loaded {len(self.anime_data)} anime records")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data file: {e}") from None
            
        # Verify data structure
        if not all(key in self.anime_data[0] for key in ['anime_id', 'Name', 'Genres']):
            raise ValueError("Loaded data is missing required fields")
            
    
    
    def get_recommendations(self, anime_title: str) -> List[Dict]:
        """
        Get anime recommendations using a two-stage pipeline:
        1. First, get genre-based similarity using Jaccard similarity
        2. Then apply SVD-based content similarity
        """
        # Step 1: Get genre-based similarity
        genre_sim = GenreJaccardSim(self.anime_data, self.anime_to_index)
        genre_sim_output = genre_sim.get_similar_anime(anime_title)
        
        # Early exit if no genre results
        if not genre_sim_output:
            return []

        # Step 2: Apply SVD directly on genre results
        try:
            # Exclude last item which is the query anime itself
            svd_input = genre_sim_output[:-1]  
            
            # Need at least 2 items for SVD to work
            if len(svd_input) >= 2:
                svd_processor = Svd(svd_input)
                return svd_processor.process_recs()
            
            return svd_input  # Return what we have if not enough for SVD
            
        except Exception as e:
            print(f"SVD failed: {e}, returning genre results")
            return sorted(svd_input, key=lambda x: x['similarity'], reverse=True)
        
    

    def get_recommendations_from_description(self, description: str) -> List[Dict]:
        """
        Direct SVD-based recommendations from description
        Bypasses genre and sentiment steps
        """
        # Create temporary anime entry
        temp_anime = {
            "Name": "[QUERY]",
            "Synopsis": description,
            "anime_id": -1
        }
        
        # Create modified dataset with temp entry
        modified_data = self.anime_data.copy()
        modified_data.append(temp_anime)
        
        # Process directly with SVD
        svd_processor = Svd(modified_data, temp_anime['Name'])
        return svd_processor.process_recs()

# Example usage for testing
if __name__ == "__main__":

    # Initialize the recommender
    recommender = AnimeRecommender()
    anime_title = "Solo Leveling"
    anime_descr = "dark fantasy with funny vibes"
    
    # Get recommendations for the specified anime
    # recommendations = recommender.get_recommendations(anime_title)
    recommendations = recommender.get_recommendations_from_description(anime_descr)
    print(len(recommendations))
    
    # Print the top 10 recommendations with their similarity scores
    print(f"\nTop 10 recommendations for {anime_title}:")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"\n{i}. {rec['Name']}")
        print(f"   Similarity: {rec['similarity']:.3f}")
         