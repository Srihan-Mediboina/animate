import json
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF:
    def __init__(self, anime_data: List[Dict]):
        """
        Initialize the TFIDF class with necessary mapping files.
        
        """
        self.anime_data = anime_data
        # Initialize TF-IDF vectorizer
        self.vectorizer = self.build_vectorizer(
            max_features=4000,
            stop_words='english',
            
        )

    @staticmethod
    def build_vectorizer(max_features: int, stop_words: str, max_df: float = 0.8, min_df: int = 10, norm: str = 'l2') -> TfidfVectorizer:
        """Returns a TfidfVectorizer object with the above preprocessing properties.
        
        Note: This function may log a deprecation warning. This is normal, and you
        can simply ignore it.
        
        Parameters
        ----------
        max_features : int
            Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer 
            constructer.
        stop_words : str
            Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer. 
        max_df : float
            Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer. 
        min_df : float
            Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer. 
        norm : str
            Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer. 

        Returns
        -------
        TfidfVectorizer
            A TfidfVectorizer object with the given parameters as its preprocessing properties.
        """
        return TfidfVectorizer(
            max_df=max_df,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            norm=norm
        )

    def process_synopses(self) -> List[str]:
        """Process the synopses from anime data.
        
        Parameters
        ----------
        anime_data : List[Dict]
            List of dictionaries containing anime information
            
        Returns
        -------
        List[str]
            List of processed synopses
        """
        # Extract and process synopses
        synopses = []
        for anime in self.anime_data:
            # Clean the synopsis text
            synopsis = anime.get('Synopsis', '')
            # Replace escaped quotes and newlines
            synopsis = synopsis.replace('\\"', ' ').replace('\\n\\n', ' ').replace('\\n', ' ')
            synopses.append(synopsis)
        return synopses

    

    def fit_transform_synopses(self) -> np.ndarray:
        """Fit the vectorizer and transform the synopses in one step.
        
        Parameters
        ----------
        vectorizer : TfidfVectorizer
            The vectorizer to fit and use for transformation
        anime_data : List[Dict]
            List of dictionaries containing anime information
            
        Returns
        -------
        np.ndarray
            Transformed synopses
        """
        # Process synopses once
        synopses = self.process_synopses()
        
        # Fit and transform in one step
        return self.vectorizer.fit_transform(synopses)

    def calculate_cosine_similarity_with_last_row(self, vectors: np.ndarray) -> Dict[int, float]:
        """Calculate cosine similarity between each row and the last row.
        
        Parameters
        ----------
        vectors : np.ndarray
            Matrix of TF-IDF vectors where each row represents a document
            
        Returns
        -------
        Dict[int, float]
            Dictionary where keys are row indices and values are cosine similarities
            with the last row
        """
        # Get the last row as the reference vector
        reference_vector = vectors[-1].toarray().flatten()
        
        # Initialize dictionary to store similarities
        similarities = {}
        
        # Calculate cosine similarity for each row with the last row
        for i in range(vectors.shape[0] - 1):  # Exclude the last row itself
            current_vector = vectors[i].toarray().flatten()
            
            # Calculate cosine similarity
            dot_product = np.dot(current_vector, reference_vector)
            norm_current = np.linalg.norm(current_vector)
            norm_reference = np.linalg.norm(reference_vector)
            
            # Avoid division by zero
            if norm_current == 0 or norm_reference == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_current * norm_reference)
            
            similarities[i] = float(similarity)
        
        return similarities

    def get_recommendations(self, similarities: Dict[int, float]) -> List[Dict]:
        """
        Get recommendations for an anime based on cosine similarity.
        
        Args:
            similarities (Dict[int, float]): The cosine similarities between the anime data
            
        Returns:
            List[Dict]: The sorted anime data
        """
        # Create a copy of anime data to avoid modifying the original
        sorted_anime = self.anime_data.copy()
        
        # Add similarity scores to each anime
        for idx, similarity in similarities.items():
            sorted_anime[idx]['similarity'] = similarity
            
        # Sort by similarity score
        return sorted(sorted_anime, key=lambda x: x.get('similarity', 0), reverse=True)

    def process_recs(self) -> List[Dict]:
        """
        Process the recommendations from the anime data.
        
        Returns:
            List[Dict]: The processed anime data with similarity scores
        """
        try:
            # Transform synopses to TF-IDF vectors
            tfidf_vectors = self.fit_transform_synopses()
            
            # Calculate similarities
            similarities = self.calculate_cosine_similarity_with_last_row(tfidf_vectors)
            
            # Get sorted recommendations
            return self.get_recommendations(similarities)
        except Exception as e:
            print(f"Error in process_recs: {e}")
            return []
