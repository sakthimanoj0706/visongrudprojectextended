import numpy as np

def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Computes the Cosine Similarity between two face embedding vectors.
    
    Similarity = (a . b) / (||a|| * ||b||)
    For L2-normalized embeddings, this simplifies to the dot product (a . b).
    
    Args:
        embedding_a (np.ndarray): Embedding vector (1D or 2D).
        embedding_b (np.ndarray): Embedding vector (1D or 2D).
        
    Returns:
        float: Cosine similarity score in range [-1.0, 1.0].
    """
    # Flatten arrays to 1D vectors
    vec_a = embedding_a.flatten()
    vec_b = embedding_b.flatten()
    
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    # Standard cosine similarity formula
    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return float(similarity)

def is_match(similarity: float, threshold: float = 0.60) -> bool:
    """
    Checks if the cosine similarity score meets or exceeds the threshold.
    
    Args:
        similarity (float): The calculated cosine similarity.
        threshold (float): Match threshold.
        
    Returns:
        bool: True if it's a match, False otherwise.
    """
    return similarity >= threshold
