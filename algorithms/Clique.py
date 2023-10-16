import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.csgraph
import math
from pandas import read_csv

from sklearn import metrics
from ast import literal_eval

class Cluster:
    def __init__(self, id, dense_units, dimensions, data_point_ids):
        self.id = id
        self.dense_units = dense_units
        self.dimensions = dimensions
        self.data_point_ids = data_point_ids

    def __str__(self):
        return "Dense units: " + str(self.dense_units.tolist()) + "\nDimensions: " \
               + str(self.dimensions) + "\nCluster size: " + str(len(self.data_point_ids)) \
               + "\nData points:\n" + str(self.data_point_ids) + "\n"


# Inserts joined item into candidates list only if its dimensionality fits
def insert_if_join_condition(candidates, item, item2, current_dim):
    joined = item.copy()
    joined.update(item2)
    if (len(joined.keys()) == current_dim) & (not candidates.__contains__(joined)):
        candidates.append(joined)


# Prune all candidates, which have a (k-1) dimensional projection not in (k-1) dim dense units
def prune(candidates, prev_dim_dense_units):
    for c in candidates:
        if not subdims_included(c, prev_dim_dense_units):
            candidates.remove(c)


def subdims_included(candidate, prev_dim_dense_units):
    for feature in candidate:
        projection = candidate.copy()
        projection.pop(feature)
        if not prev_dim_dense_units.__contains__(projection):
            return False
    return True


def self_join(prev_dim_dense_units, dim):
    candidates = []
    for i in range(len(prev_dim_dense_units)):
        for j in range(i + 1, len(prev_dim_dense_units)):
            insert_if_join_condition(
                candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], dim)
    return candidates


def is_data_in_projection(tuple, candidate, xsi):
    for feature_index, range_index in candidate.items():
        feature_value = tuple[feature_index]
        if int(feature_value * xsi % xsi) != range_index:
            return False
    return True


def get_dense_units_for_dim(data, prev_dim_dense_units, dim, xsi, tau, verbose):
    candidates = self_join(prev_dim_dense_units, dim)
    prune(candidates, prev_dim_dense_units)

    # Count number of elements in candidates
    projection = np.zeros(len(candidates))
    number_of_data_points = np.shape(data)[0]
    for dataIndex in range(number_of_data_points):
        for i in range(len(candidates)):
            if is_data_in_projection(data[dataIndex], candidates[i], xsi):
                projection[i] += 1

    # Return elements above density threshold
    is_dense = projection > tau * number_of_data_points
    if verbose:
        print("projection: ", projection)
        print("is_dense: ", is_dense)
    return np.array(candidates)[is_dense]


def build_graph_from_dense_units(dense_units):
    graph = np.identity(len(dense_units))
    for i in range(len(dense_units)):
        for j in range(len(dense_units)):
            graph[i, j] = get_edge(dense_units[i], dense_units[j])
    return graph


def get_edge(node1, node2):
    dim = len(node1)
    distance = 0

    if node1.keys() != node2.keys():
        return 0

    for feature in node1.keys():
        distance += abs(node1[feature] - node2[feature])
        if distance > 1:
            return 0

    return 1


def get_cluster_data_point_ids(data, cluster_dense_units, xsi):
    point_ids = set()

    # Loop through all dense unit
    for u in cluster_dense_units:
        tmp_ids = set(range(np.shape(data)[0]))
        # Loop through all dimensions of dense unit
        for feature_index, range_index in u.items():
            tmp_ids = tmp_ids & set(
                np.where(np.floor(data[:, feature_index] * xsi % xsi) == range_index)[0])
        point_ids = point_ids | tmp_ids

    return point_ids


def get_clusters(dense_units, data, xsi, verbose):
    graph = build_graph_from_dense_units(dense_units)
    number_of_components, component_list = scipy.sparse.csgraph.connected_components(
        graph, directed=False)

    dense_units = np.array(dense_units)
    clusters = []
    # For every cluster
    for i in range(number_of_components):
        # Get dense units of the cluster
        cluster_dense_units = dense_units[np.where(component_list == i)]
        if verbose:
            print("cluster_dense_units: ", cluster_dense_units.tolist())

        # Get dimensions of the cluster
        dimensions = set()
        for u in cluster_dense_units:
            dimensions.update(u.keys())

        # Get points of the cluster
        cluster_data_point_ids = get_cluster_data_point_ids(
            data, cluster_dense_units, xsi)
        # Add cluster to list
        clusters.append(Cluster(i, cluster_dense_units,
                                dimensions, cluster_data_point_ids))

    return clusters


def get_one_dim_dense_units(data, tau, xsi, verbose):
    number_of_data_points = np.shape(data)[0]
    number_of_features = np.shape(data)[1]
    # store number of data points in each grid
    projection = np.zeros((xsi, number_of_features))
    for f in range(number_of_features):
        for element in data[:, f]:
            projection[int(element * xsi % xsi), f] += 1
    is_dense = projection > tau * number_of_data_points
    if verbose:
        print("1D projection:\n", projection, "\n")
        print("is_dense:\n", is_dense)
    one_dim_dense_units = []
    for f in range(number_of_features):
        for unit in range(xsi):
            if is_dense[unit, f]:
                dense_unit = dict({f: unit})
                one_dim_dense_units.append(dense_unit)
    return one_dim_dense_units


# Normalize data in all features (1e-5 padding is added because clustering works on [0,1) interval)
def normalize_features(data):
    normalized_data = data
    number_of_features = np.shape(normalized_data)[1]
    for f in range(number_of_features):
        normalized_data[:, f] -= min(normalized_data[:, f]) - 1e-5
        normalized_data[:, f] *= 1 / (max(normalized_data[:, f]) + 1e-5)
    return normalized_data


def evaluate_clustering_performance(clusters, labels):
    set_of_dimensionality = set()
    for cluster in clusters:
        set_of_dimensionality.add(frozenset(cluster.dimensions))

    # Evaluating performance in all dimensionality
    for dim in set_of_dimensionality:
        print("\nEvaluating clusters in dimension: ", list(dim))
        # Finding clusters with same dimensions
        clusters_in_dim = []
        for c in clusters:
            if c.dimensions == dim:
                clusters_in_dim.append(c)
        clustering_labels = np.zeros(np.shape(labels))
        for i, c in enumerate(clusters_in_dim):
            clustering_labels[list(c.data_point_ids)] = i + 1

        print("Number of clusters: ", len(clusters_in_dim))
        print("Adjusted Rand index: ", metrics.adjusted_rand_score(
            labels, clustering_labels))
        print("Adjusted Mutual Information: ", metrics.adjusted_mutual_info_score(
            labels, clustering_labels))

        print("Homogeneity, completeness, V-measure: ",
              metrics.homogeneity_completeness_v_measure(labels, clustering_labels))

        print("Fowlkes-Mallows: ",
              metrics.fowlkes_mallows_score(labels, clustering_labels))
        
def evaluate_clique_unsupervised(clusters, data, file_name=''):
    """
    evaluation function for data without true labels
    only evaluate clusters in the largest subspace
    scores used: Silhoutte score, Calinski-Harabasz index, Davies-Bouldin Index

    :param clusters: a list of Clusters
    :param data: 2d numpy array
    :param file_name: name of the file to write the results, 
        default is empty string

    No return
    """

    # Evaluating performance of largest subspace
    f = open(file_name+'_evaluation.txt', 'a')

    clustering_labels = clique_clusters_to_labels(clusters)

    f.write("Number of clusters: "+str(len(clusters)))
    f.write('\n')
    f.write("Silhoutte coefficient: "+str(metrics.silhouette_score(data, clustering_labels)))
    f.write('\n')
    f.write("Calinski-Harabasz Index: "+str(metrics.calinski_harabasz_score(data, clustering_labels)))
    f.write('\n')
    f.write("Davies-Bouldin Index: "+str(metrics.davies_bouldin_score(data, clustering_labels)))
    f.write('\n')
    
    f.close()

def evaluate_clique_supervised(clusters, true_labels, file_name=''):
    """
    evaluation function for data with true labels
    only evaluate clusters in the largest subspace
    scores used: ARI, AMI, V-measure

    :param clusters: a list of Clusters
    :param true_labels: 1d numpy array, true clustering labels
    :param file_name: name of the file to write the results, 
        default is empty string

    No return
    """

    # Evaluating performance of largest subspace
    f = open(file_name+'_evaluation.txt', 'a')

    clustering_labels = clique_clusters_to_labels(clusters)

    f.write("Number of clusters: "+str(len(clusters)))
    f.write('\n')
    f.write("Adjusted Rand Index: "+str(metrics.adjusted_rand_score(true_labels, clustering_labels)))
    f.write('\n')
    f.write("Adjusted Mutual Information: "+str(metrics.adjusted_mutual_info_score(true_labels, clustering_labels)))
    f.write('\n')
    f.write("Homogeneity, completeness, V-measure: "+str(metrics.homogeneity_completeness_v_measure(true_labels, clustering_labels)))
    f.write('\n')
    
    f.close()
        
def clique_clusters_to_labels(clusters, N):
    """
    Find the largest subspace of CLIQUE clustering result, and turn that into clustering labels

    :param clusters: return of run_clique
    :param N: total number of data points

    :return clustering_labels: 1d numpy array containing label of clusters
    """

    highest_dimensions = 0
    for cluster in clusters:
        if len(cluster.dimensions) > highest_dimensions:
            largest_subspace = cluster.dimensions
            highest_dimensions = len(cluster.dimensions)

    clusters_in_largest_dim = []
    for c in clusters:
            if c.dimensions == largest_subspace:
                clusters_in_largest_dim.append(c)

    clustering_labels = np.zeros(N)
    for i, c in enumerate(clusters_in_largest_dim):
        clustering_labels[list(c.data_point_ids)] = i + 1

    return clustering_labels

def run_clique(data, xsi, tau, verbose=False):
    """
    Main function for CLIQUE clustering algorithm

    :param data: 2d numpy array, each row is a data point
    :param xsi: int, number of grids in each dimension
    :param tau: float [0, 1], selectivity threshold 
        a grid containing more than tau proportion of data will be treated as a dense unit
    """

    # Finding 1 dimensional dense units
    dense_units = get_one_dim_dense_units(data, tau, xsi, verbose)

    # Getting 1 dimensional clusters
    clusters = get_clusters(dense_units, data, xsi, verbose)

    # Finding dense units and clusters for dimension > 2
    current_dim = 2
    number_of_features = np.shape(data)[1]
    while (current_dim <= number_of_features) & (len(dense_units) > 0):
        if verbose:
            print("\n", str(current_dim), " dimensional clusters:")
        dense_units = get_dense_units_for_dim(
            data, dense_units, current_dim, xsi, tau, verbose)
        for cluster in get_clusters(dense_units, data, xsi, verbose):
            clusters.append(cluster)
        current_dim += 1

    return clusters


def read_labels(delimiter, label_column, path):
    return np.genfromtxt(path, dtype="U10", delimiter=delimiter, usecols=[label_column])


def read_data(delimiter, feature_columns, path):
    return np.genfromtxt(path, dtype=float, delimiter=delimiter, usecols=feature_columns)

def plot_clusters(data, clusters, title, xsi):
    # Check if there are clusters to plot
    if len(clusters) <= 0:
        return

    ndim = data.shape[1]
    nrecords = data.shape[0]
    data_extent = [[min(data[:, x]), max(data[:, x])] for x in range(0, ndim)]
    plt_nrow = math.floor(ndim ** 0.5)
    plt_ncol = plt_nrow * (1 - plt_nrow) + ndim
    plt_cmap = plt.cm.tab10
    plt_marker_size = 10
    plt_spacing = 0  # change spacing to apply a margin to data_extent

    # Plot clusters in each dimension
    for dim in range(1, ndim + 1):
        # Get all clusters in 'dim' dimension(s)
        clusters_in_dim = []
        for c in clusters:
            if len(c.dimensions) == dim:
                clusters_in_dim.append(c)

        # Check if there are clusters in 'dim' dimension(s)
        dim_nclusters = len(clusters_in_dim)
        if dim_nclusters <= 0:
            continue

        # subplot for the current dimension (dim)
        ax = plt.subplot(plt_nrow, plt_ncol, dim)

        # Plot all data points as black points
        if dim == 1:
            ax.scatter(data[:, 0], [0] * nrecords,
                       s=plt_marker_size, c=["black"], label="noise")
            ax.scatter([0] * nrecords, data[:, 1],
                       s=plt_marker_size, c=["black"])
        elif dim == 2:
            ax.scatter(data[:, 0], data[:, 1],
                       s=plt_marker_size, c=["black"], label="noise")

        # For all clusters in 'dim' dimension(s)
        for i, c in enumerate(clusters_in_dim):
            c_size = len(c.data_point_ids)
            c_attrs = list(c.dimensions)
            c_elems = list(c.data_point_ids)

            if dim == 1:  # one-dimensional clusters
                x = data[c_elems, 0] if c_attrs[0] == 0 else [0] * c_size
                x = np.array(x)
                y = data[c_elems, 1] if c_attrs[0] == 1 else [0] * c_size
                y = np.array(y)
            elif dim == 2:  # two-dimensional clusters
                x = data[c_elems, c_attrs[0]]
                y = data[c_elems, c_attrs[1]]
            
            print(c.id)
            ax.scatter(x, y, s=plt_marker_size, c=[
                plt_cmap(c.id)], label=str(c.id))

        ax.set_xlim(data_extent[0][0] - plt_spacing,
                    data_extent[0][1] + plt_spacing)
        ax.set_ylim(data_extent[1][0] - plt_spacing,
                    data_extent[1][1] + plt_spacing)
        ax.set_title(str(dim) + "-dimensional clusters")
        ax.legend(title="Cluster ID")

        # Putting grids on the charts
        minor_ticks_x = np.linspace(
            data_extent[0][0], data_extent[0][1], xsi + 1)
        minor_ticks_y = np.linspace(
            data_extent[1][0], data_extent[1][1], xsi + 1)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(visible=True, which="minor", axis="both")

    plt.gcf().suptitle(title)
    plt.show()



if __name__ == "__main__":
    print("this is a file containing the codes for CLIQUE algorithm")