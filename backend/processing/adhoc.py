import numpy as np
from .svd import Svd
from typing import List, Dict, Any, Optional
import re
import copy
class AdHoc:
    def __init__(self, anime_data: List[Dict[str, Any]], n_components: int = 60):
        """
        Initialize the AdHoc recommendation system.
        
        Args:
            anime_data: List of anime dictionaries containing all anime information
            n_components: Number of components for SVD (default: 60)
        """
        self.anime_data = copy.deepcopy(anime_data)
        self.n_components = n_components
        self.svd = None

    def _parse_episode_range(self, range_str: str) -> tuple:
        """Parse episode range string into min and max values."""
        # Handles different ranges based on input
        ranges = {
            "1001-4000": (1001, 4000),
            "201-1000": (201, 1000),
            "101-200": (101, 200),
            "61-100": (61, 100),
            "31-60": (31, 60),
            "21-30": (21, 30),
            "11-20": (11, 20),
            "1-10": (1, 10),
        }
        return ranges.get(range_str, (0, float('inf')))

    def _filter_by_episodes(self, anime: Dict[str, Any], selected_ranges: List[str]) -> float:
        """Calculate episode match score (0-1)."""
        if not selected_ranges:
            return 1.0  # No episode filter means perfect match
            
        try:
            # Handle case where Episodes might be a float or string
            episodes = float(anime.get('Episodes', 0))
            episodes = int(episodes) if episodes.is_integer() else episodes  # Handle both int and float representations
        except (ValueError, TypeError):
            return 0.0
            
        if episodes == 0:
            return 0.0
            
        best_match = 0.0
        for range_str in selected_ranges:
            min_ep, max_ep = self._parse_episode_range(range_str)
            if min_ep <= episodes <= max_ep:
                range_size = max_ep - min_ep
                if range_size == 0:
                    return 1.0
                position = (episodes - min_ep) / range_size
                match_score = 1 - abs(position - 0.5) * 2
                best_match = max(best_match, match_score)
                
        return best_match

    def _filter_by_genres(self, anime: Dict[str, Any], selected_genres: List[str]) -> float:
        """Calculate genre match score (0-1)."""
        if not selected_genres:
            return 1.0  # No genre filter means perfect match
            
        anime_genres = [g.strip() for g in anime.get('Genres', '').split(',')]
        if not anime_genres:
            return 0.0
            
        matches = sum(1 for genre in selected_genres if genre in anime_genres)
        return matches / len(selected_genres)

    def _filter_by_studios(self, anime: Dict[str, Any], selected_studios: List[str]) -> float:
        """Calculate studio match score (0-1)."""
        if not selected_studios:
            return 1.0  # No studio filter means perfect match
            
        anime_studios = [s.strip() for s in anime.get('Studios', '').split(',')]
        if not anime_studios:
            return 0.0
            
        matches = sum(1 for studio in selected_studios if studio in anime_studios)
        return matches / len(selected_studios)

    def _filter_by_rating(self, anime: Dict[str, Any], selected_ratings: List[str]) -> float:
        """Calculate rating match score (0-1)."""
        if not selected_ratings:
            return 1.0  # No rating filter means perfect match
            
        anime_rating = anime.get('Rating', '')
        if not anime_rating:
            return 0.0
            
        return 1.0 if anime_rating in selected_ratings else 0.0

    def get_recommendations(
    self,
    genres: Optional[List[str]] = None,
    episodes: Optional[List[str]] = None,
    studios: Optional[List[str]] = None,
    ratings: Optional[List[str]] = None,
    description: Optional[str] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
      """
      Get recommendations based on filters and description.
      
      Args:
          genres: List of selected genres
          episodes: List of selected episode ranges
          studios: List of selected studios
          ratings: List of selected ratings
          description: Text description of desired anime
          top_n: Number of recommendations to return
          
      Returns:
          List of recommended anime with similarity scores
      """
      # Initialize filters if None
      genres = genres or []
      episodes = episodes or []
      studios = studios or []
      ratings = ratings or []
      
    #   print(f"\nProcessing filters:")
    #   print(f"Genres: {genres}")
    #   print(f"Episodes: {episodes}")
    #   print(f"Studios: {studios}")
    #   print(f"Ratings: {ratings}")
    #   print(f"Description: {description}\n")
      
      # First pass: Filter and score based on criteria
      filtered_anime = []
      for anime in self.anime_data:
          # Check if all filters match (AND condition)
          episode_score = self._filter_by_episodes(anime, episodes)
          genre_score = self._filter_by_genres(anime, genres)
          studio_score = self._filter_by_studios(anime, studios)
          rating_score = self._filter_by_rating(anime, ratings)
          
          # Include anime only if all conditions pass (AND condition)
          if episode_score > 0 and genre_score > 0 and studio_score > 0 and rating_score > 0:
              # Calculate overall filter score (weighted average)
              filter_scores = [episode_score, genre_score, studio_score, rating_score]
              filter_score = sum(filter_scores) / len(filter_scores)  # Simple average, can adjust weights if needed
              
              anime['filter_score'] = filter_score
              filtered_anime.append(anime)
      
      print(f"Found {len(filtered_anime)} anime matching filter criteria")
      
      # If no description provided, return filtered results
      if not description:
          print("No description provided, returning filtered results")
          return sorted(filtered_anime, key=lambda x: x['filter_score'], reverse=True)
      
      # Second pass: Use SVD for semantic similarity with description
      print("Applying SVD for semantic similarity with description")
      target_anime = {"Synopsis": description}
      self.svd = Svd(filtered_anime, target_anime, n_components=self.n_components)
      svd_recommendations = self.svd.process_recs()
      
      if not svd_recommendations:
          print("No SVD recommendations found, falling back to filtered results")
          return sorted(filtered_anime, key=lambda x: x['filter_score'], reverse=True)
      
      print(f"Found {len(svd_recommendations)} SVD recommendations")
      
      # Combine filter scores with SVD similarity scores
      for anime in svd_recommendations:
          
              # Combine scores: 70% SVD similarity, 30% filter match
            anime['similarity'] = max(0.55 * anime['similarity'] + 0.45 * anime['filter_score'], anime['similarity'])
      
      print("Returning combined SVD and filter recommendations")
      return sorted(svd_recommendations, key=lambda x: x['similarity'], reverse=True)
