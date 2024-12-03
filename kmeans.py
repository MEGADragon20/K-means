from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import math
import numpy as np

exampleCenters = 4
exampleNSamples = 75
x_y, c = make_blobs(cluster_std = 0.4, n_samples= exampleNSamples, centers= exampleCenters)

def abstand_berechnen(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def k_means(cluster_centers, cluster_count, points):
    if not cluster_centers:
        cluster_centers = [[np.random.uniform(min(points[:, 0]), max(points[:, 0])),
                            np.random.uniform(min(points[:, 1]), max(points[:, 1]))]
                           for _ in range(cluster_count)]
    
    for i in range(100):
        clusters = [[] for _ in range(cluster_count)]
        
        # Assign points to the nearest cluster
        for point in points:
            distances = [abstand_berechnen(point, center) for center in cluster_centers]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        
        # Update cluster centers
        new_cluster_centers = []
        for cluster in clusters:
            if cluster:  # Handle non-empty cluster
                cluster_array = np.array(cluster)
                new_center = cluster_array.mean(axis=0).tolist()
            else:  # Handle empty cluster (fallback to random point)
                new_center = [np.random.uniform(min(points[:, 0]), max(points[:, 0])),
                              np.random.uniform(min(points[:, 1]), max(points[:, 1]))]
            new_cluster_centers.append(new_center)
        
        # Check for convergence
        center_shifts = [abstand_berechnen(old, new) for old, new in zip(cluster_centers, new_cluster_centers)]
        if max(center_shifts) < 0.0001:
            break
        
        cluster_centers = new_cluster_centers  
    
    return cluster_centers, clusters
def avg_avg_len_of_clusters(centers, clusters):
    avg_distances = 0
    for i in range(len(centers)):
        distances = 0
        for j in clusters[i]:
            distances += abstand_berechnen(centers[i], j)
        avg_distances += distances/len(clusters[i])
    return avg_distances/len(centers)

        


def elbow(points, constant = 0.2):
    a = 1
    while True:
        current_centers, current_clusters = k_means([], a, points)
        gradient_this_point = avg_avg_len_of_clusters(current_centers, current_clusters)/a
        print(a, gradient_this_point)
        if gradient_this_point < constant:
            return current_centers, current_clusters
        a += 1


    


final_centers, final_clusters = elbow(x_y)

plt.figure(figsize=(8, 6))
for cluster in final_clusters:
    cluster_array = np.array(cluster)
    plt.scatter(cluster_array[:, 0], cluster_array[:, 1])

for center in final_centers:
    plt.scatter(center[0], center[1], c='red', marker='x', s=100)

plt.title("K-Means Clustering")
plt.show()


        

