import json
from typing import List, Dict
from scipy.sparse.linalg import svds
import numpy as np
from .tfidf import TFIDF

class Svd:
    def __init__(self, anime_data: List[Dict], n_components: int = 100):
        """
        Initialize the SVD class with anime data and SVD parameters.
        
        Args:
            anime_data: List of dictionaries containing anime information
            n_components: Number of dimensions for SVD reduction
        """
        self.anime_data = anime_data
        self.n_components = n_components
        self.tfidf = TFIDF(anime_data)
        self.reduced_vectors = self._perform_svd()

    def _perform_svd(self) -> np.ndarray:
        """Perform SVD on TF-IDF matrix and return reduced vectors."""
        # Get TF-IDF matrix from TFIDF processor
        tfidf_matrix = self.tfidf.fit_transform_synopses()
        
        # Perform SVD and sort singular values in descending orderd
        u, s, vt = svds(tfidf_matrix, k=self.n_components)
        sorted_idx = np.argsort(s)[::-1]
        u = u[:, sorted_idx]
        s = s[sorted_idx]
        
        # Create reduced dimension vectors
        return u @ np.diag(s)

    def calculate_cosine_similarity_with_last_row(self) -> Dict[int, float]:
        """Calculate cosine similarity between each row and the last row in reduced space."""
        reference_vector = self.reduced_vectors[-1]
        similarities = {}
        
        for i in range(len(self.reduced_vectors) - 1):
            current_vector = self.reduced_vectors[i]
            dot_product = np.dot(current_vector, reference_vector)
            norm_current = np.linalg.norm(current_vector)
            norm_reference = np.linalg.norm(reference_vector)
            
            similarity = dot_product / (norm_current * norm_reference) if (norm_current * norm_reference) != 0 else 0
            similarities[i] = float(similarity)
            
        return similarities

    def get_recommendations(self, similarities: Dict[int, float]) -> List[Dict]:
        """Sort anime based on similarity scores (same as TFIDF version)."""
        sorted_anime = self.anime_data.copy()
        for idx, similarity in similarities.items():
            sorted_anime[idx]['similarity'] = similarity
        return sorted(sorted_anime, key=lambda x: x.get('similarity', 0), reverse=True)

    def process_recs(self) -> List[Dict]:
        """Main processing pipeline for SVD recommendations."""
        try:
            similarities = self.calculate_cosine_similarity_with_last_row()
            return self.get_recommendations(similarities)
        except Exception as e:
            print(f"Error in SVD process_recs: {e}")
            return []
   