import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
import time
import warnings

from performance_evaluator import evaluate_clusters_supervised, joint_metric
from data_generator import dataGen

from pyclustering.cluster.bang import bang, bang_visualizer
from algorithms.Clique import normalize_features, run_clique, clique_clusters_to_labels
from algorithms.waveCluster import waveCluster
from algorithms.optigrid import Optigrid

def clusters_to_labels(clusters, N):
    """
    Convert cluster representation to labels
    0 means noise

    :param clusters: list of numpy arrays, each contains indices of data points in that cluster
    :param N: total number of data points

    :return clustering_labels: 1d numpy array of labels. same length as data set
    """

    clustering_labels = np.zeros(N)
    for i, cluster in enumerate(clusters):
        clustering_labels[cluster] = i + 1

    return clustering_labels

def parameter_tuning(process, data, parameter_choices, labels, plot=False):
    """
    using the joint metric, tune hyperparameters of a clustering algorithm
    grid search will be performed

    :param process: python function, the clustering process for tuning
        must only take TWO argument, data, tuple of parameters
        return clustering labels
    :param data: 2d numpy array of data points
    :param labels: true clustering labels
    :param parameter_choices: list of 1d numpy arrays, each array contain choices for that parameter
    :param plot: if true, plot the score against parameters
        for plotting purpose, each item in parameter_choices must be sorted in order

    :return best_parameters: tuple of best parameters
    """

    shape = []
    indice_dict = []
    for l in parameter_choices:
        num = l.shape[0]
        shape.append(num)
        indice_dict.append(np.arange(num))

    # store grid of indices
    indices = []
    for index in product(*indice_dict):
        indices.append(index)

    # number of parameters
    num_param = len(shape)
    scores = np.zeros(shape)

    print(f"Tuning {num_param} parameter(s) for {process.__name__}")

    # perform grid search and store the scores
    for i, parameters in enumerate(product(*parameter_choices)):
        clustering_labels = process(data, parameters)
        score = joint_metric(labels, clustering_labels)
        scores[indices[i]] = score

    # find the maximum score
    argmax = np.argmax(scores)
    max_index = indices[argmax]
    best_parameters = ()
    for i, index in enumerate(max_index):
        parameter = parameter_choices[i][index]
        # detect whether best parameter at margin
        if index == 0 or index == shape[i]-1:
            print(f"WARNING: location of parameter {i+1} with best score is at the margin with value {parameter}, please check the plot for details")
            plot = True
        best_parameters += (parameter,)

    if plot:
        # 1-dimensional plot
        if num_param == 1:
            plt.title(f'Hyperparameter Tuning for {process.__name__}')
            plt.xlabel('parameter')
            plt.ylabel('score')
            plt.plot(parameter_choices[0], scores)
            plt.savefig(f'Tuning_Graph_{process.__name__}', bbox_inches='tight')
            plt.show()
        # 2-dimensional plot
        elif num_param == 2:
            ax = sns.heatmap(scores, vmax=1, yticklabels=parameter_choices[0], xticklabels=parameter_choices[1])
            ax.set(xlabel='parameter 2', ylabel='parameter 1')
            plt.savefig(f'Tuning_Graph_{process.__name__}', bbox_inches='tight')
            plt.show()
        else:
            print("Plot is only supported for 1 or 2 parameter(s)")

    return best_parameters

def BANG_process(data, parameters, plot=False):
    levels = parameters[0]
    bang_instance = bang(data=data, levels=levels)
    bang_instance.process()
    bang_clusters = bang_instance.get_clusters()
    bang_labels = clusters_to_labels(bang_clusters, data.shape[0])

    if plot:
        noise = bang_instance.get_noise()
        bang_visualizer.show_clusters(data, bang_clusters, noise)

    return bang_labels

def CLIQUE_process(data, parameters):
    xsi = parameters[0]
    tau = parameters[1]
    clique_clusters = run_clique(data, xsi, tau)
    clique_labels = clique_clusters_to_labels(clique_clusters, data.shape[0])

    return clique_labels

def wave_process(data, parameters):
    threshold = parameters[0]
    return waveCluster(data, threshold=threshold)

def opti_process(data, parameters, noise_level=0, q=1, suppress_warning=True, seed=2022):
    if suppress_warning:
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    max_cut_score = parameters[0]
    kde_bandwidth = parameters[1]
    np.random.seed(seed)
    optigrid = Optigrid(d=data.shape[1], q=q, max_cut_score=max_cut_score, kde_bandwidth=kde_bandwidth, noise_level=noise_level, ignore=True)
    # use uniform weight
    optigrid.fit(data, weights = np.ones(data.shape[0]))
    optigrid_labels = clusters_to_labels(optigrid.clusters, data.shape[0])

    warnings.filterwarnings('default')

    return optigrid_labels

dim = 2
N = 5000
generator = dataGen(N, dim, k=10, seed=2022)

original_data, labels = generator.use_shape('ring', separation=3)
labels = labels.reshape(-1)
plt.title('Plot of first two axes of the data')
plt.scatter(original_data[:, 0], original_data[:, 1], s=1, c=labels)
plt.savefig('data_set')
plt.show()

data = normalize_features(original_data)


# BANG parameter tuning
levels_choices = [np.arange(11, 18)]
bang_parameters = parameter_tuning(BANG_process, data, levels_choices, labels=labels, plot=True)

st = time.time()
bang_labels = BANG_process(data, bang_parameters)
et = time.time()
execution_time = et - st
evaluate_clusters_supervised(labels, bang_labels, dim, time=execution_time, file_name='BANG'+'_ring')


# CLIQUE tuning
# number of grids in each dimension
xsi_choice = np.arange(22, 29)
# selectivity threshold
tau_choice = np.arange(0, 0.01, 0.01)
clique_parameter_choices = [xsi_choice, tau_choice]
clique_parameters = parameter_tuning(CLIQUE_process, data, clique_parameter_choices, labels)

st = time.time()
clique_labels = CLIQUE_process(data, clique_parameters)
et = time.time()
execution_time = et - st
evaluate_clusters_supervised(labels, clique_labels, dim, time=execution_time, file_name='CLIQUE'+'_ring')

# waveCluster tuning
threshold_choice = [np.arange(-5, 3, 0.5)]
wave_parameters = parameter_tuning(wave_process, data, threshold_choice, labels, plot=True)

st = time.time()
wavelet_labels = wave_process(data, wave_parameters)
et = time.time()
execution_time = et - st
evaluate_clusters_supervised(labels, wavelet_labels, dim, time=execution_time, file_name='waveCluster'+'_ring')

# Optigrid tuning
max_cut_score_choice = np.arange(0.4, 0.8, 0.05)
bandwidth_choice = np.arange(0.01, 0.1, 0.01)
opti_choices = [max_cut_score_choice, bandwidth_choice]
opti_parameters = parameter_tuning(opti_process, data, opti_choices, labels, plot=True)

st = time.time()
opti_labels = opti_process(data, opti_parameters)
et = time.time()
execution_time = et - st
evaluate_clusters_supervised(labels, opti_labels, dim, time=execution_time, file_name='optigrid'+'_ring')