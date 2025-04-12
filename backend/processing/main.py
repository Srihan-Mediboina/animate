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
from .genre_jaccard_sim import GenreJaccardSim
from .reviewer_sentiment import ReviewerSentiment
from .svd import Svd

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
    
    def __init__(self):
        """
        Initialize the AnimeRecommender.
        No initialization is needed as all data is loaded at module level.
        """
        pass
        
    def get_recommendations(self, anime_title: str) -> List[Dict]:
        """
        Get anime recommendations using a multi-stage pipeline approach.
        
        The pipeline works as follows:
        1. First, get genre-based similarity using Jaccard similarity
        2. Then, refine these recommendations using reviewer sentiment analysis
        3. Finally, apply SVD-based content similarity for the most relevant matches
        
        Args:
            anime_title (str): Title of the anime to get recommendations for
            
        Returns:
            List[Dict]: List of recommended anime with their scores, sorted by similarity
                       Each dict contains the anime's details and similarity score
        """
        # Step 1: Get genre-based similarity
        genre_sim = GenreJaccardSim(anime_data, anime_to_index)
        genre_sim_output = genre_sim.get_similar_anime(anime_title)
        
        # Step 2: Refine using reviewer sentiment analysis
        reviewer_sentiment = ReviewerSentiment(anime_to_reviewer, reviewer_to_anime, anime_to_index, index_to_anime_id)
        reviewer_sentiment_output = reviewer_sentiment.get_highly_rated_anime(anime_title, genre_sim_output)
        
        # Step 3: Apply SVD-based content similarity
        svd_processor = Svd(reviewer_sentiment_output)
        svd_output = svd_processor.process_recs()
        
        # Fallback logic: If SVD returns no results, use previous stage's results
        if len(svd_output) == 0:
            if len(reviewer_sentiment_output) == 0:
                # If no reviewer sentiment results, use genre similarity results
                print("returning genre_sim_output")
                return sorted(genre_sim_output[:-1], key=lambda x: x['similarity'], reverse=True)
            else:
                # If we have reviewer sentiment results, use those
                print("returning reviewer_sentiment_output sim", reviewer_sentiment_output[-1]['similarity'])
                return reviewer_sentiment_output
        else:
            # Use SVD results if available
            return svd_output

# Example usage for testing
if __name__ == "__main__":
    # Initialize the recommender
    recommender = AnimeRecommender()
    anime_title = "Attack on Titan"
    
    # Get recommendations for the specified anime
    recommendations = recommender.get_recommendations(anime_title)
    
    # Print the top 10 recommendations with their similarity scores
    print(f"\nTop 10 recommendations for {anime_title}:")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"\n{i}. {rec['Name']}")
        print(f"   Similarity: {rec['similarity']:.3f}")
         