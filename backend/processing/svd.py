import json
from typing import List, Dict
from scipy.sparse.linalg import svds
import numpy as np
from .tfidf import TFIDF

class Svd:
    def __init__(self, anime_data: List[Dict], target_anime_data: Dict, n_components: int = 60):
        """
        Initialize the SVD class with anime data and SVD parameters.
        
        Args:
            anime_data: List of dictionaries containing anime information
            query_anime: The anime title to find recommendations for
            n_components: Number of dimensions for SVD reduction
        """
        try:
        
            self.anime_data = anime_data
            self.target_anime = target_anime_data
            self.n_components = n_components
            self.tfidf = TFIDF(anime_data, target_anime_data)
        except Exception as e:
            print(f"Error in SVD initialization: {e}")
            raise

    # def _find_query_index(self) -> int:
    #     """Find the index of the query anime in the data."""
    #     for i, anime in enumerate(self.anime_data):
    #         if anime['Name'].lower() == self.query_anime.lower():
    #             return i
    #     raise ValueError(f"Anime '{self.query_anime}' not found in data")

    def _perform_svd(self) -> np.ndarray:
        """Perform SVD on TF-IDF matrix and return reduced vectors."""
        try:
            # Get TF-IDF matrix from TFIDF processor
            tfidf_matrix = self.tfidf.fit_transform_synopses()
            
            # Ensure matrix is large enough for SVD
            min_dim = min(tfidf_matrix.shape)
            if min_dim <= 1:
                raise ValueError(f"Matrix too small for SVD. Shape: {tfidf_matrix.shape}")
                return []
                
            # Ensure n_components is valid
            if self.n_components >= min_dim:
                self.n_components = min(min_dim - 1, 50)  # Use at most 50 components
                print(f"Adjusted n_components to {self.n_components} based on matrix shape {tfidf_matrix.shape}")
            
            # Perform SVD and sort singular values in descending order
            u, s, vt = svds(tfidf_matrix, k=self.n_components)
            sorted_idx = np.argsort(s)[::-1]
            u = u[:, sorted_idx]
            s = s[sorted_idx]
            
            # Create reduced dimension vectors
            return u @ np.diag(s)
        except Exception as e:
            print(f"Error in SVD calculation: {e}")
            raise

    def calculate_cosine_similarity_with_query(self) -> Dict[int, float]:
        """Calculate cosine similarity between each row and the query anime in reduced space."""
        try:
            reduced_vectors = self._perform_svd()
            reference_vector = reduced_vectors[-1]
            similarities = {}
            
            for i in range(len(reduced_vectors)-1):
                current_vector = reduced_vectors[i]
                dot_product = np.dot(current_vector, reference_vector)
                norm_current = np.linalg.norm(current_vector)
                norm_reference = np.linalg.norm(reference_vector)
                
                similarity = dot_product / (norm_current * norm_reference) if (norm_current * norm_reference) != 0 else 0
                similarities[i] = float(similarity)
                
            return similarities
        except Exception as e:
            print(f"Error in cosine similarity calculation: {e}")
            raise

    def get_recommendations(self, similarities: Dict[int, float]) -> List[Dict]:
        """Sort anime based on similarity scores (same as TFIDF version)."""
        sorted_anime = self.anime_data.copy()
        for idx, similarity in similarities.items():
            sorted_anime[idx]['similarity'] = similarity
        return sorted(sorted_anime, key=lambda x: x.get('similarity', 0), reverse=True)

    def process_recs(self) -> List[Dict]:
        """Main processing pipeline for SVD recommendations."""
        try:
            similarities = self.calculate_cosine_similarity_with_query()
            return self.get_recommendations(similarities)
        except Exception as e:
            print(f"Error in SVD process_recs: {e}")
            return []
   