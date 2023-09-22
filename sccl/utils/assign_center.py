import numpy as np

def assign_to_closest_centers(embeddings, cluster_centers, n=3):
    """
    embedding 1,2,3 Closest
    embedding 1,2,3 Next Closest
    embedding 1,2,3 Next Next Closest
    """
    distances = np.linalg.norm(embeddings[:, np.newaxis] - cluster_centers, axis=2)
    
    # Find the indices of the n closest centers for each embedding
    assignments = np.argsort(distances, axis=1)[:, :n]

    # Get the distances to the n closest centers for each embedding
    closest_distances = np.take_along_axis(distances, assignments, axis=1)

    assignments_ordered = []
    closest_distances_ordered = []

    # Order them consecutively for each embedding
    for i in range(n):
        assignments_ordered.extend(assignments[:, i])
        closest_distances_ordered.extend(closest_distances[:, i])
    
    return np.array(assignments_ordered), np.array(closest_distances_ordered)
