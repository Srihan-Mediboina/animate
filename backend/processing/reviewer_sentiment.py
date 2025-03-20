import json
from typing import List, Dict
import re


class ReviewerSentiment:
    def __init__(self, anime_to_reviewer: Dict[str, List], reviewer_to_anime: Dict[str, List], anime_to_index: Dict[str, int], index_to_anime_id: Dict[str, int]):
        """
        Initialize the ReviewerSentiment class with necessary data files.
        """
        self.anime_to_reviewer = anime_to_reviewer
        self.reviewer_to_anime = reviewer_to_anime
        self.anime_to_index = anime_to_index
        self.index_to_anime_id = index_to_anime_id
    
    def get_highly_rated_anime(self, anime_title: str, intermediate_results: List[Dict]) -> List[Dict]:
        """
        Filter intermediate results to only include anime with high reviewer ratings (i.e 9+).
        
        Args:
            anime_title (str): Title of the query anime
            intermediate_results (List[Dict]): Results from genre similarity filtering
            
        Returns:
            List[Dict]: Filtered list of anime with high reviewer ratings, preserving similarity scores
        """
        if not intermediate_results:
            return []
        
        # Get the index of the query anime
        query_anime_index = self.anime_to_index.get(anime_title)
        if query_anime_index is None:
            return []
        
        # Get the id of the query anime
        query_anime_id = self.index_to_anime_id.get(str(query_anime_index))
        if query_anime_id is None:
            return []
        
        print("query_anime_id", query_anime_id)
        # Get the reviewers who highly rated the query anime
        query_reviewers = self.anime_to_reviewer.get(str(query_anime_id), [])

        jaccard_set = set()

        for anime in intermediate_results:
            anime_id = self.index_to_anime_id.get(str(self.anime_to_index.get(anime['Name'])))
            if anime_id is not None:
                jaccard_set.add(anime_id)
        print("jaccard_set", len(jaccard_set))
        
        highly_rated_anime = set()
       
        print("query_reviewers", len(query_reviewers))
        for reviewer in query_reviewers:
            reviewer_anime_ids = self.reviewer_to_anime.get(str(reviewer), [])
            highly_rated_anime.update(reviewer_anime_ids)
        
        print("highly_rated_anime", len(highly_rated_anime))
        
        final_ids = jaccard_set.intersection(highly_rated_anime)
        print("final_ids", len(final_ids))
        
        final_results = [anime for anime in intermediate_results if int(anime["anime_id"])in final_ids]

        return final_results


# if __name__ == "__main__":
#     # Import GenreJaccardSim
#     from genre_jaccard_sim import GenreJaccardSim
    
#     # Create instances of both classes
#     genre_sim = GenreJaccardSim(anime_data, anime_to_index)
#     reviewer_sentiment = ReviewerSentiment(anime_to_reviewer, reviewer_to_anime, anime_to_index, index_to_anime_id)
    
#     # Test with a sample anime title
#     test_anime = "Attack on Titan"
#     print(f"\nTesting recommendation pipeline for: {test_anime}")
    
#     # Step 1: Get genre-similar anime using Jaccard similarity
#     print("\nStep 1: Finding genre-similar anime...")
#     similar_animes = genre_sim.get_similar_anime(test_anime)
#     print(f"Found {len(similar_animes)} anime with similar genres")
#     # print("\nFirst 5 genre-similar anime:")
#     # for anime in similar_animes[:5]:
#     #     print(f"\nName: {anime['Name']}")
#     #     print(f"Genres: {anime['Genres']}")
    
#     # Step 2: Filter by reviewer sentiment
#     print("\nStep 2: Filtering by reviewer sentiment...")
#     highly_rated = reviewer_sentiment.get_highly_rated_anime(test_anime, similar_animes)
#     print(f"\nFound {len(highly_rated)} highly rated similar anime")
#     print("\nFinal recommendations:")
#     print(highly_rated[-1]['Name'])