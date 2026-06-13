# clustering.py
import math

def calculate_cosine_distance(vec_a, vec_b):
    """Computes distance based on spatial vector orientation."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    # Since our vectors are already normalized to unit length, 
    # Cosine Similarity is simply the dot product.
    # Distance = 1 - Similarity
    distance = 1.0 - dot_product
    return max(0.0, distance)  # Prevent micro floating-point rounding errors below 0

class AgglomerativeClusteringFromScratch:
    def __init__(self):
        self.history = []

    def fit(self, matrix):
        """Executes a bottom-up merge strategy until one root cluster remains."""
        num_docs = len(matrix)
        # Initialize clusters: each document starts inside its own isolated list
        clusters = {i: [i] for i in range(num_docs)}
        
        # Build the initial pairwise distance matrix between individual documents
        distance_matrix = {}
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                dist = calculate_cosine_distance(matrix[i], matrix[j])
                distance_matrix[(i, j)] = dist

        cluster_counter = num_docs
        
        # Sequentially merge clusters until only one master cluster remains
        while len(clusters) > 1:
            min_dist = float('inf')
            pair_to_merge = None
            
            cluster_ids = list(clusters.keys())
            
            # Find the two closest separate clusters using Complete Linkage
            for idx, c1 in enumerate(cluster_ids):
                for c2 in cluster_ids[idx + 1:]:
                    max_linkage_dist = -1.0
                    
                    # Complete Linkage: find maximum distance between any element of c1 and c2
                    for doc_a in clusters[c1]:
                        for doc_b in clusters[c2]:
                            # Look up precalculated base distances
                            lookup_key = (min(doc_a, doc_b), max(doc_a, doc_b))
                            dist = distance_matrix[lookup_key]
                            if dist > max_linkage_dist:
                                max_linkage_dist = dist
                                
                    if max_linkage_dist < min_dist:
                        min_dist = max_linkage_dist
                        pair_to_merge = (c1, c2)
            
            c1, c2 = pair_to_merge
            
            # Log the structural merge history tracking tree construction
            self.history.append({
                'merged_nodes': (c1, c2),
                'new_node_id': cluster_counter,
                'linkage_distance': min_dist,
                'combined_elements': clusters[c1] + clusters[c2]
            })
            
            # Form a new unified cluster node
            clusters[cluster_counter] = clusters[c1] + clusters[c2]
            
            # Delete old individual cluster references from memory mapping
            del clusters[c1]
            del clusters[c2]
            
            cluster_counter += 1
            
        return self.history

    def display_tree(self, total_documents):
        """Prints a simplified visual log layout representing tree branches."""
        print("\n=== Hierarchical Tree Building Sequence ===")
        for step, entry in enumerate(self.history):
            print(f"Step {step + 1}: Combined Cluster {entry['merged_nodes'][0]} and "
                  f"Cluster {entry['merged_nodes'][1]} into Cluster {entry['new_node_id']} "
                  f"(Linkage Dist: {entry['linkage_distance']:.4f})")
