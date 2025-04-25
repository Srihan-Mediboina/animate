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
    
    def get_highly_rated_anime(self, target_anime_data: Dict, intermediate_results: List[Dict]) -> List[Dict]:
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
        
        
        # Get the id of the query anime
        query_anime_id = int(target_anime_data['anime_id'])
        if query_anime_id is None:
            return []
        
        # Get the reviewers who highly rated the query anime
        query_reviewers = self.anime_to_reviewer.get(str(query_anime_id), [])

        jaccard_set = set()

        for anime in intermediate_results:
            anime_id = int(anime['anime_id'])
            if anime_id is not None:
                jaccard_set.add(anime_id)

        
        highly_rated_anime = set()
       
        for reviewer in query_reviewers:
            reviewer_anime_ids = self.reviewer_to_anime.get(str(reviewer), [])
            highly_rated_anime.update(reviewer_anime_ids)
        
        
        final_ids = jaccard_set.intersection(highly_rated_anime) - {query_anime_id}
        
        
        final_results = [anime for anime in intermediate_results if int(anime["anime_id"]) in final_ids]
        
        return final_results

