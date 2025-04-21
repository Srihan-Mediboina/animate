import json
import numpy as np
from typing import List, Dict, Set


class GenreJaccardSim:
    def __init__(self, anime_data: List[Dict], anime_to_index: Dict[str, int]):
        """
        Initialize the GenreJaccardSim class with necessary data files.
        
        """
        self.anime_data = anime_data
        self.anime_to_index = anime_to_index
        
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1 (Set): First set
            set2 (Set): Second set
            
        Returns:
            float: Jaccard similarity score
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def get_similar_anime(self, anime_title: str, threshold: float = 0.45) -> List[Dict]:
        """
        Get anime with high genre similarity to the input anime.
        
        Args:
            anime_title (str): Title of the anime to find similar ones for
            threshold (float): Minimum Jaccard similarity score (default: 0.2)
            
        Returns:
            List[Dict]: List of similar anime with their similarity scores
        """
        if anime_title not in self.anime_to_index:
            print(f"Anime '{anime_title}' not found in index")
            return []
        
        target_idx = self.anime_to_index[anime_title]
        target_anime_genres = set(genre.strip() for genre in self.anime_data[target_idx]['Genres'].split(','))
        
        similar_animes = []
        
        for anime in self.anime_data:
            if anime['Name'] != anime_title:  # Exclude the input anime title
                anime_genres = set(genre.strip() for genre in anime['Genres'].split(','))
                similarity = self.jaccard_similarity(target_anime_genres, anime_genres)
                
                if similarity >= threshold:
                    # Make a copy of the anime data to avoid modifying the original
                    anime_copy = anime.copy()
                    anime_copy['similarity'] = similarity
                    similar_animes.append(anime_copy)
        
        return (similar_animes, self.anime_data[target_idx])
