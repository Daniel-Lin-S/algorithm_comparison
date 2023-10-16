import numpy as np
import sklearn.metrics as metrics

def evaluate_clusters(data, clustering_labels, file_name=''):
    """
    Evaluate clusters without ground truth, 
    using Silhoutte coefficient, Calinski-Harabasz Index, Davies-Bouldin Index
    and write that into a file

    :param clustering_labels: 1d numpy array of predicted clustering labels
    :param data: 2d numpy array, each row is a data point
    :param file_name: name of the file to write the results, 
        default is empty string
    """

    f = open(file_name+'_evaluation.txt', 'a')

    num_clusters = len(set(clustering_labels))
    n_samples = np.shape(data)[0]
    if num_clusters < 2:
        raise Exception('Only one cluster, not possible to evaluate')
    elif num_clusters > n_samples - 1:
        raise Exception('Too many clusters, not possible to evaluate')

    f.write("Number of clusters: "+str(num_clusters))
    f.write('\n')
    f.write("Silhoutte coefficient: "+str(metrics.silhouette_score(data, clustering_labels)))
    f.write('\n')
    f.write("Calinski-Harabasz Index: "+str(metrics.calinski_harabasz_score(data, clustering_labels)))
    f.write('\n')
    f.write("Davies-Bouldin Index: "+str(metrics.davies_bouldin_score(data, clustering_labels)))
    f.write('\n')
    
    f.close()

def evaluate_clusters_supervised(true_labels, clustering_labels, dim, time=None, file_name=""):
    """
    Evaluate clusters with ground truth, 
    using ARI, AMI, V-measure 
    and write that into a file

    :param true_labels: 1d numpy array of ground truth labels of clusters
    :param clustering_labels: 1d numpy array of predicted clustering labels
    :param dim: dimension of data
    :param time: execution time in seconds
    :param file_name: name of the file to write the results, 
        default is empty string
    """

    f = open('evaluation/'+file_name+'.txt', 'a')
    f.write('---------Experiment---------')
    f.write('\n')
    f.write(f'Number of data points: {true_labels.shape[0]}')
    f.write('\n')
    f.write(f'Number of Dimensions: {dim}')
    f.write('\n')
    if time:
        f.write(f'Execution Time: {time}s')
        f.write('\n')

    labels = set(clustering_labels)
    num_clusters = len(labels)

    # remove small clusters
    for label in labels:
        if np.sum(clustering_labels == label) < 5:
            num_clusters -= 1
    
    # discard noise
    keep_ind = true_labels != -1
    true_labels = true_labels[keep_ind]
    clustering_labels = np.array(clustering_labels)[keep_ind]
    f.write("Number of clusters recognised: "+str(num_clusters))
    f.write('\n')
    f.write("Adjusted Rand Index: "+str(metrics.adjusted_rand_score(true_labels, clustering_labels)))
    f.write('\n')
    f.write("Adjusted Mutual Information: "+str(metrics.adjusted_mutual_info_score(true_labels, clustering_labels)))
    f.write('\n')
    f.write("Homogeneity, completeness, V-measure: "+str(metrics.homogeneity_completeness_v_measure(true_labels, clustering_labels)))
    f.write('\n')
    
    f.close()

def joint_metric(true_labels, clustering_labels):
    """
    average of V-measure, AMI, ARI

    :param true_labels: 1d numpy array of ground truth labels of clusters
    :param clustering_labels: 1d numpy array of predicted clustering labels

    :return joint_metric: average of three scores
    """

    # discard noise 
    keep_ind = true_labels != -1
    true_labels = true_labels[keep_ind]
    clustering_labels = np.array(clustering_labels)[keep_ind]
    
    return (metrics.adjusted_rand_score(true_labels, clustering_labels) + 
            metrics.adjusted_mutual_info_score(true_labels, clustering_labels) + 
            metrics.v_measure_score(true_labels, clustering_labels)) / 3 