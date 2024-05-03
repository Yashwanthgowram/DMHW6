"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial.distance import pdist, squareform

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """

    true_labels=labels
    def density_weighted_adjacency_matrix(data, k, threshold):
        distance_matrix = squareform(pdist(data, 'euclidean'))
        neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
        n = len(data)
        adjacency_matrix = np.zeros((n, n), dtype=bool)
        local_density = np.array([np.sum(np.exp(-distance_matrix[i, neighbors[i]])) for i in range(n)])
        normalized_density = local_density / np.max(local_density)

        for i in range(n):
            for j in range(i + 1, n):
                mutual_density = min(normalized_density[i], normalized_density[j])
                shared_neighbors_count = len(np.intersect1d(neighbors[i], neighbors[j]))
                if shared_neighbors_count >= k * 0.5 and mutual_density > threshold:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True

        return adjacency_matrix

    def calculate_sse(data, labels, cluster_centers):
        if data is None or len(data) == 0 or labels is None or cluster_centers is None:
            return 0.0
        sse = 0
        for k in range(len(cluster_centers)):
            cluster_data = data[labels == k]
            if cluster_data.size == 0:
                continue
            sse += np.sum((cluster_data - cluster_centers[k])**2)
        return sse
    
    def modified_rand(true_classes, pred_clusters):
        # Find unique true classes and predicted clusters
        unique_classes = np.unique(true_classes)
        unique_clusters = np.unique(pred_clusters)

        # Create a contingency table
        contingency_table = np.zeros((unique_classes.size, unique_clusters.size), dtype=int)
        for class_idx, class_label in enumerate(unique_classes):
            for cluster_idx, cluster_label in enumerate(unique_clusters):
                contingency_table[class_idx, cluster_idx] = np.sum((true_classes == class_label) & (pred_clusters == cluster_label))

        # Compute row and column sums
        row_sums = np.sum(contingency_table, axis=1)
        col_sums = np.sum(contingency_table, axis=0)

        # Compute the number of pairs and their combinations
        n_pairs = sum([n_ij * (n_ij - 1) / 2 for n_ij in contingency_table.flatten()])
        row_pairs = sum([n_ij * (n_ij - 1) / 2 for n_ij in row_sums])
        col_pairs = sum([n_ij * (n_ij - 1) / 2 for n_ij in col_sums])

        # Compute terms for the adjusted Rand index
        n_samples = true_classes.size
        total_pairs = n_samples * (n_samples - 1) / 2
        expected_index = row_pairs * col_pairs / total_pairs
        max_index = (row_pairs + col_pairs) / 2
        denominator = (max_index - expected_index)

        # Handle the special case when the denominator is 0
        if denominator == 0:
            return 1 if n_pairs == expected_index else 0

        modified_rand = (n_pairs - expected_index) / denominator

        return modified_rand
    
    def custom_dbscan(similarity_matrix, data_points, min_points):
        num_points = similarity_matrix.shape[0]
        cluster_assignments = -np.ones(num_points)
        cluster_id = 0
        cluster_centers = []
        for i in range(num_points):
            if cluster_assignments[i] != -1:
                continue
            neighbors = np.where(similarity_matrix[i])[0]
            if len(neighbors) < min_points:
                cluster_assignments[i] = -2
                continue
            cluster_assignments[i] = cluster_id
            seed_set = set(neighbors)
    
            cluster_points = [data_points[i]]
    
            while seed_set:
                current_point = seed_set.pop()
                cluster_points.append(data_points[current_point])
                if cluster_assignments[current_point] == -2:
                    cluster_assignments[current_point] = cluster_id
                if cluster_assignments[current_point] != -1:
                    continue
                cluster_assignments[current_point] = cluster_id
                current_neighbors = np.where(similarity_matrix[current_point])[0]
                if len(current_neighbors) >= min_points:
                    seed_set.update(current_neighbors)
    
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
            cluster_id += 1
        return cluster_assignments, np.array(cluster_centers)
    
    
    adjacency_matrix = density_weighted_adjacency_matrix(data, k=params_dict['k'], threshold=2)
    labels,cluster_assignments=custom_dbscan(adjacency_matrix, data, min_points=params_dict['smin'])
    
    sse = calculate_sse(data, labels, cluster_assignments)

    ari = modified_rand(true_labels, labels)

    computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    answers = {}
    data = np.load("C:/Users/gyash/OneDrive/Desktop/erlebacherHW6/CAP-5771-s24-hw6-main/question1_cluster_data.npy")
    true_labels=np.load("C:/Users/gyash/OneDrive/Desktop/erlebacherHW6/CAP-5771-s24-hw6-main/question1_cluster_labels.npy")

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    sse_final=[]
    preds_final=[]
    ari_final=[]
    eigen_final=[]
    
    i=0
    while i<5 and len(true_labels) >= (i + 1) * 1000:
        datav = data[i * 1000:(i + 1) * 1000]
        true_labelsv = true_labels[i * 1000:(i + 1) * 1000]

        params_dict = {'k': 5, 'smin': 5}
        preds, sse_hyp, ari_hyp = jarvis_patrick(datav, true_labelsv, params_dict)
        if sse_hyp is None:
            sse_hyp = 0.0 

        sse_final.append(sse_hyp)
        ari_final.append(ari_hyp if ari_hyp is not None else 0.0)
        preds_final.append(preds)

        if i not in groups:
            groups[i] = {'k': 5, 'smin': 5, 'ARI': ari_hyp, 'SSE': sse_hyp}
        i+=1
    sse_numpy = np.array(sse_final)
    ari_numpy = np.array(ari_final)
    


    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    least_sse_index=np.argmin(sse_numpy)
    highest_ari_index=np.argmax(ari_numpy)
    lowest_ari_index=np.argmin(ari_numpy)

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter(data[1000*highest_ari_index:(highest_ari_index+1)*1000, 0], data[1000*highest_ari_index:(highest_ari_index+1)*1000, 1], c=preds_final[highest_ari_index] if np.any(preds_final[highest_ari_index]) else [0] * len(data[1000*highest_ari_index:(highest_ari_index+1)*1000]), cmap='viridis', marker='.')
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    plt.title('Largest ARI')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    plt.grid(True)
    plt.savefig("LargestARI_JP.png")


    plot_SSE = plt.scatter(data[1000*least_sse_index:(least_sse_index+1)*1000, 0], data[1000*least_sse_index:(least_sse_index+1)*1000, 1], c=preds_final[least_sse_index] if np.any(preds_final[least_sse_index]) else [0] * len(data[1000*least_sse_index:(least_sse_index+1)*1000]), cmap='viridis', marker='.')
    # plt.scatter(true_labelsv[:, 0], true_labelsv[:, 1], c=datav, cmap='viridis', marker='.')
    plt.title('Least SSE')
    plt.xlabel(f'Feature 1 for Dataset{i+1}')
    plt.ylabel(f'Feature 2 for Dataset{i+1}')
    plt.grid(True)
    plt.savefig("SmallestSSE_JP.png")
    plt.close()



    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    ARI_sum=[]
    SSE_sum=[]
    for i in groups:
      if 'ARI' in groups[i]:
        ARI_sum.append(groups[i]['ARI'])
        SSE_sum.append(groups[i]['SSE'])

    # A single float
    answers["mean_ARIs"] = float(np.mean(ari_numpy))
    # print(type(float(np.mean(np.array(ARI_sum)))))

    # A single float
    answers["std_ARIs"] = float(np.std(ari_numpy))
    # print(type(np.std(np.array(ARI_sum))))

    # A single float
    answers["mean_SSEs"] = float(np.mean(sse_numpy))

    # A single float
    answers["std_SSEs"] = float(np.std(sse_numpy))

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
