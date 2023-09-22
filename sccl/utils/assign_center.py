import numpy as np
from scipy.spatial import distance

def assign_to_closest_center(embeddings, cluster_centers):
    """
    Assigns each embedding to the closest cluster center and returns the distance.

    Parameters:
    - embeddings: numpy array of shape (num_embeddings, embedding_dim)
    - cluster_centers: numpy array of shape (num_clusters, embedding_dim)

    Returns:
    - assignments: numpy array of shape (num_embeddings,) with the index of the closest center for each embedding
    - closest_distances: numpy array of shape (num_embeddings,) with the distance to the closest center for each embedding
    """

    # Compute pairwise distances
    distances = distance.cdist(embeddings, cluster_centers, 'euclidean')

    # Find the index of the closest center for each embedding
    assignments = np.argmin(distances, axis=1)

    # Get the distance to the closest center for each embedding
    closest_distances = distances[np.arange(distances.shape[0]), assignments]

    return assignments, closest_distances
