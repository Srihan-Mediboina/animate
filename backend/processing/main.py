"""
Main module for the Anime Recommendation System.
This module implements a hybrid recommendation system that combines multiple approaches:
1. Genre-based similarity using Jaccard similarity
2. Reviewer sentiment analysis
3. TF-IDF based content similarity

The system uses a pipeline approach where each step refines the recommendations
from the previous step to provide more accurate and relevant suggestions.
"""

import os
import json
from typing import List, Dict
from .genre_jaccard_sim import GenreJaccardSim
from .reviewer_sentiment import ReviewerSentiment
from .tfidf import TFIDF
from ..data_loader import DataLoader

# Initialize the data loader
data_loader = DataLoader()

# Load all required data
try:

    # Load the mapping of anime titles to their indices in the dataset
    # This is used for quick lookups when processing recommendations
    anime_to_index = data_loader.load_json_file('anime_to_index.json')

    # Load the reviewer-anime relationship data
    # anime_to_reviewer contains a map of anime id to a list of ids of reviewers who gave it a score of 9+
    # reviewer_to_anime contains a map of reviewer id to a list of anime ids they rated highly
    # These mappings help in finding similar anime based on reviewer preferences
    anime_to_reviewer = data_loader.load_json_file('anime_to_reviewer.json')
    reviewer_to_anime = data_loader.load_json_file('reviewer_to_anime.json')

    # Load the mapping of indices to anime IDs
    # This helps in maintaining consistency between different data representations
    # and is used to convert between internal indices and MyAnimeList IDs
    index_to_anime_id = data_loader.load_json_file('index_to_anime_id.json')
except Exception as e:
    print(f"Error loading data: {e}")
    anime_data = []
    anime_to_index = {}
    anime_to_reviewer = {}
    reviewer_to_anime = {}
    index_to_anime_id = {}

class AnimeRecommender:
    """
    Main class for the anime recommendation system.
    Implements a hybrid recommendation approach combining multiple similarity metrics.
    """
    
    def __init__(self, anime_data=None):
        """Initialize the recommender with pre-loaded data or load it if not provided."""
        
        if anime_data is not None:
            # Use pre-loaded data
            self.anime_data = anime_data
            
        
    def get_recommendations(self, anime_title: str) -> List[Dict]:
        """
        Get anime recommendations using a multi-stage pipeline approach.
        
        The pipeline works as follows:
        1. First, get genre-based similarity using Jaccard similarity
        2. Then, refine these recommendations using reviewer sentiment analysis
        3. Finally, apply TF-IDF based content similarity for the most relevant matches
        
        Args:
            anime_title (str): Title of the anime to get recommendations for
            
        Returns:
            List[Dict]: List of recommended anime with their scores, sorted by similarity
                       Each dict contains the anime's details and similarity score
        """
        # Step 1: Get genre-based similarity
        genre_sim = GenreJaccardSim(self.anime_data, anime_to_index)
        genre_sim_output = genre_sim.get_similar_anime(anime_title)
        
        # Step 2: Refine using reviewer sentiment analysis
        reviewer_sentiment = ReviewerSentiment(anime_to_reviewer, reviewer_to_anime, anime_to_index, index_to_anime_id)
        reviewer_sentiment_output = reviewer_sentiment.get_highly_rated_anime(anime_title, genre_sim_output)
        
        # Step 3: Apply TF-IDF based content similarity
        tfidf = TFIDF(reviewer_sentiment_output)
        tfidf_output = tfidf.process_recs()
        
        # Fallback logic: If TF-IDF returns no results, use previous stage's results
        if len(tfidf_output) == 0:
            if len(reviewer_sentiment_output) == 0:
                # If no reviewer sentiment results, use genre similarity results
                print("returning genre_sim_output")
                return sorted(genre_sim_output[:-1], key=lambda x: x['similarity'], reverse=True)
            else:
                # If we have reviewer sentiment results, use those
                print("returning reviewer_sentiment_output sim", reviewer_sentiment_output[-1]['similarity'])
                return reviewer_sentiment_output
        else:
            # Use TF-IDF results if available
            return tfidf_output

# # Example usage for testing
# if __name__ == "__main__":
#     # Initialize the recommender
#     recommender = AnimeRecommender()
#     anime_title = "Attack on Titan"
    
#     # Get recommendations for the specified anime
#     recommendations = recommender.get_recommendations(anime_title)
    
#     # Print the top 10 recommendations with their similarity scores
#     print(f"\nTop 10 recommendations for {anime_title}:")
#     for i, rec in enumerate(recommendations[:10], 1):
#         print(f"\n{i}. {rec['Name']}")
#         print(f"   Similarity: {rec['similarity']:.3f}")
         