import sys
import numpy as np
import scipy.io
import scipy.io.wavfile


# get a string line for the output file with the iterations count and the centroids
def get_iter_line(iter_count, centroids):
    return f"[iter {iter_count}]:{','.join([str(i) for i in centroids])}\n"


# get the rounded average of all the points in the given cluster
def average_of_points(cluster):
    points_sum = np.zeros(len(cluster[0]))
    for point in cluster:
        points_sum = np.add(points_sum, point)
    average = np.divide(points_sum, len(cluster))
    return np.around(average)


# update all the centroids to the average of the points in their cluster
def update_centroids(clusters, centroids):
    size = len(centroids)
    for i in range(size):
        cluster = clusters[i]
        if cluster:
            centroids[i] = average_of_points(cluster)


# calculate the distance between 2 points
def points_distance(point1, point2):
    subtraction = np.subtract(point1, point2)
    norm = np.linalg.norm(subtraction)
    return pow(norm, 2)


# find the closest centroid to the given point
def closest_centroid(point, centroids):
    min_distance = float('inf')
    min_index = -1
    size = len(centroids)
    for i in range(size):
        distance = points_distance(point, centroids[i])
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index


# assign each point in x to the cluster of its closest centroid
def assign_points(x, clusters, centroids):
    for point in x:
        centroid_index = closest_centroid(point, centroids)
        if centroid_index != -1:
            clusters[centroid_index].append(point)
    return clusters


# initialize k empty clusters
def init_clusters(k):
    clusters = []
    for i in range(k):
        clusters.append([])
    return clusters


# the k-means algorithm as learned in class, with 30 iterations max
# returns the full string for the output file
def k_means(k, x, centroids):
    output_string = ""
    iter_count = 0
    iter_max = 30
    convergence = False
    while not convergence and iter_count < iter_max:
        clusters = init_clusters(k)
        clusters = assign_points(x, clusters, centroids)
        old_centroids = centroids.copy()
        update_centroids(clusters, centroids)
        convergence = np.array_equal(centroids, old_centroids)
        output_string += get_iter_line(iter_count, centroids)
        iter_count += 1
    return output_string


# the main function of the program
def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    k = len(centroids)
    output_string = k_means(k, x, centroids)
    output_file = open("output.txt", "w")
    output_file.write(output_string)
    output_file.close()


main()
 
