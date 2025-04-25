import json
from typing import List, Dict, Tuple
from scipy.sparse.linalg import svds
import numpy as np
from .tfidf import TFIDF

class Svd:
    def __init__(self, anime_data: List[Dict], target_anime_data: Dict, n_components: int = 100):
        """
        Initialize the SVD class with anime data and SVD parameters.
        
        Args:
            anime_data: List of dictionaries containing anime information
            target_anime_data: The anime to find recommendations for
            n_components: Number of dimensions for SVD reduction
        """
        try:
            self.anime_data = anime_data
            self.target_anime = target_anime_data
            self.n_components = n_components
            self.tfidf = TFIDF(anime_data, target_anime_data)
            self.u = None  # Left singular vectors
            self.s = None  # Singular values
            self.vt = None  # Right singular vectors (rows = components, columns = words)
            self.feature_names = None  # Words from TF-IDF
        except Exception as e:
            print(f"Error in SVD initialization: {e}")
            raise

    def _perform_svd(self) -> np.ndarray:
        """Perform SVD on TF-IDF matrix and store components."""
        try:
            # Get TF-IDF matrix and feature names
            tfidf_matrix = self.tfidf.fit_transform_synopses()
            self.feature_names = self.tfidf.get_feature_names_out()
            
            # Ensure matrix is large enough for SVD
            min_dim = min(tfidf_matrix.shape)
            if min_dim <= 1:
                raise ValueError(f"Matrix too small for SVD. Shape: {tfidf_matrix.shape}")
                
            # Adjust n_components if necessary
            if self.n_components >= min_dim:
                self.n_components = min(min_dim - 1, 50)
                print(f"Adjusted n_components to {self.n_components} based on matrix shape {tfidf_matrix.shape}")
            
            # Perform SVD and sort singular values in descending order
            self.u, self.s, self.vt = svds(tfidf_matrix, k=self.n_components)
            sorted_idx = np.argsort(self.s)[::-1]
            self.u = self.u[:, sorted_idx]
            self.s = self.s[sorted_idx]
            self.vt = self.vt[sorted_idx, :]
            
            # Return reduced vectors
            return self.u @ np.diag(self.s)
        except Exception as e:
            print(f"Error in SVD calculation: {e}")
            raise

    def _get_top_words_for_dimension(self, dim: int, n_words: int = 5) -> List[str]:
        """Get the top words for a given SVD dimension."""
        if self.vt is None:
            raise ValueError("SVD not performed yet. Call _perform_svd() first.")
        
        # Get the weights for the specified dimension
        component_weights = self.vt[dim, :]
        
        # Sort words by their weights in this dimension
        top_word_indices = np.argsort(component_weights)[::-1][:n_words]
        
        return [self.feature_names[i] for i in top_word_indices]

    def _explain_similarity(self, anime_vector: np.ndarray, target_vector: np.ndarray, top_k_dims: int = 3) -> List[str]:
        """
        Explain why two anime are similar by finding the most important SVD dimensions
        and the top words in those dimensions.
        
        Args:
            anime_vector: Reduced vector of the anime to compare.
            target_vector: Reduced vector of the target anime.
            top_k_dims: Number of top dimensions to consider.
            
        Returns:
            List[str]: Top words contributing to similarity.
        """
        if self.vt is None:
            raise ValueError("SVD not performed yet. Call _perform_svd() first.")

        # Find dimensions where both anime have high values
        dim_importance = anime_vector * target_vector  # Element-wise product
        top_dims = np.argsort(dim_importance)[::-1][:top_k_dims]

        # Get the top words for each important dimension
        top_words = []
        for dim in top_dims:
            top_words.extend(self._get_top_words_for_dimension(dim, n_words=2))  # Get 2 words per dimension
        
        # Remove duplicates and return top 5 unique words
        unique_words = list(dict.fromkeys(top_words))  # Preserves order
        return unique_words[:5]

    def calculate_cosine_similarity_with_query(self) -> Dict[int, Tuple[float, List[str]]]:
        """
        Calculate cosine similarity between each row and the query anime in reduced space,
        along with the top 5 words contributing to the similarity.
        
        Returns:
            Dict[int, Tuple[float, List[str]]]: 
                Keys are anime indices, values are (similarity_score, top_5_words).
        """
        try:
            reduced_vectors = self._perform_svd()
            reference_vector = reduced_vectors[-1]
            similarities = {}
            
            for i in range(len(reduced_vectors)-1):
                current_vector = reduced_vectors[i]
                
                # Calculate cosine similarity
                dot_product = np.dot(current_vector, reference_vector)
                norm_current = np.linalg.norm(current_vector)
                norm_reference = np.linalg.norm(reference_vector)
                similarity = dot_product / (norm_current * norm_reference) if (norm_current * norm_reference) != 0 else 0
                
                # Get top words contributing to similarity
                top_words = self._explain_similarity(current_vector, reference_vector)
                
                similarities[i] = (float(similarity), top_words)
                
            return similarities
        except Exception as e:
            print(f"Error in cosine similarity calculation: {e}")
            raise

    def get_recommendations(self, similarities: Dict[int, any]) -> List[Dict]:
        """Sort anime based on similarity scores (same as TFIDF version)."""
        sorted_anime = self.anime_data.copy()
        for idx, similarity in similarities.items():
            sorted_anime[idx]['similarity'] = similarity[0]
            top_words_sentence = "This is similar because they share words like " + ", ".join(similarity[1])
            sorted_anime[idx]['top_words'] = top_words_sentence
        return sorted(sorted_anime, key=lambda x: x.get('similarity', 0), reverse=True)

    def process_recs(self) -> List[Dict]:
        """Main processing pipeline for SVD recommendations."""
        try:
            similarities = self.calculate_cosine_similarity_with_query()
            return self.get_recommendations(similarities)
        except Exception as e:
            print(f"Error in SVD process_recs: {e}")
            return []
   