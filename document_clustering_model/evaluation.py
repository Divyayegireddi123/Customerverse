# evaluation.py
from clustering import calculate_cosine_distance

def calculate_silhouette_score(matrix, current_clusters, file_names):
    """Calculates the Silhouette Score from scratch using Cosine Distance."""
    scores = {}
    total_score = 0.0
    num_docs = len(matrix)
    
    if len(current_clusters) <= 1 or len(current_clusters) == num_docs:
        return {name: 0.0 for name in file_names}, 0.0

    doc_to_cluster = {}
    for cluster_key, doc_indices in current_clusters.items():
        for doc_id in doc_indices:
            doc_to_cluster[doc_id] = cluster_key

    for i in range(num_docs):
        own_cluster_id = doc_to_cluster[i]
        own_cluster_indices = current_clusters[own_cluster_id]
        
        if len(own_cluster_indices) > 1:
            distances_a = [calculate_cosine_distance(matrix[i], matrix[j]) for j in own_cluster_indices if i != j]
            a_i = sum(distances_a) / len(distances_a)
        else:
            a_i = 0.0
            
        min_neighbor_distance = float('inf')
        for neighbor_cluster_id, neighbor_indices in current_clusters.items():
            if neighbor_cluster_id == own_cluster_id:
                continue
            distances_b = [calculate_cosine_distance(matrix[i], matrix[j]) for j in neighbor_indices]
            avg_b_to_cluster = sum(distances_b) / len(distances_b)
            if avg_b_to_cluster < min_neighbor_distance:
                min_neighbor_distance = avg_b_to_cluster
                
        b_i = min_neighbor_distance if min_neighbor_distance != float('inf') else 0.0
        
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
            
        scores[file_names[i]] = s_i
        total_score += s_i
        
    global_average = total_score / num_docs
    return scores, global_average
