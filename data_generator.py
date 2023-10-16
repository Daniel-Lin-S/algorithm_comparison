import mdcgenpy.clusters as mdcgen
from mdcgenpy.clusters.distributions import valid_distributions
import numpy as np
from sklearn.utils import shuffle

class dataGen:
    """
    a data generator used for simulation study
    various clusters will be generated based on mdcgen
    """

    def __init__(self, n, dim, k, noise=0, seed=None) -> None:
        """
        default setting for all data sets
        :param n: number of instances
        :param dim: dimension of data
        :param k: number of clusters
        :param noise: value in [0, 1], noise level. default 0 means no noise
        :param seed: seed for random generators default None
        """
        if n <= 0 or not isinstance(n, int): 
            raise Exception('n must be a positive integer')
        if dim <= 0 or not isinstance(dim, int): 
            raise Exception('dim must be a positive integer')
        if k <= 0 or not isinstance(k, int): 
            raise Exception('k must be a positive integer')
        if noise > 1 or noise < 0:
            raise Exception('noise must be between 0 and 1')
        
        self.n = n
        self.dim = dim
        self.k = k
        self.noise = noise
        self.seed = seed

    def generate_data(self, null_dim=0):
        """
        generate data using default setting

        :param null_dim: number of dimensions not used for clustering i.e. the useless features

        :return data: 2d numpy array, each row is a data point
        :return labels: 1d numpy array, contains labels of each data point representing the cluster memberhsip
        """

        valid_dim = self.dim - null_dim
        if valid_dim < 2:
            raise Exception('Cluster must be at least 2 dimensional')
        gen = mdcgen.ClusterGenerator(seed=self.seed, n_samples=self.n, n_feats=valid_dim, 
                                      k = self.k, outliers=np.floor(self.n*self.noise))

        data, labels = gen.generate_data()
        del gen

        data = np.c_[ data, np.random.rand(self.n, null_dim)]

        return data, labels

    def non_uniform(self, deviation):
        """
        Generate clusters which are not of uniform sizes
        cluster sizes taken from uniform distribution ranging from mean cluster size +- deviation percentage 
        note: final cluster size may slip off from n

        :param deviation: value between [0, 1], percentage standard deviation of cluster sizes

        :return data: 2d numpy array, each row is a data point
        :return labels: 1d numpy array, contains labels of each data point representing the cluster memberhsip
        """
        mean_size = np.floor(self.n/self.k)
        np.random.seed(self.seed)
        cluster_weights = np.floor(np.random.uniform(mean_size*(1-deviation), mean_size*(1+deviation), size=self.k)).astype(int)
        n_samples = np.sum(cluster_weights)

        cluster_sizes = np.random.uniform(0.1, 0.1*(1+1.3*deviation), self.k)    

        gen = mdcgen.ClusterGenerator(seed=self.seed, n_samples=n_samples, n_feats=self.dim, 
                                      k = list(cluster_weights), outliers=np.floor(self.n*self.noise), compactness_factor=list(cluster_sizes))

        data, labels = gen.generate_data()
        del gen

        return data, labels
    
    def overlapping(self, overlap):
        """
        generate clusters that could potentially overlap

        :param overlap: degree of overlap, reasonable value between 0.1 and 0.35

        :return data: 2d numpy array, each row is a data point
        :return labels: 1d numpy array, contains labels of each data point representing the cluster memberhsip
        """

        gen = mdcgen.ClusterGenerator(seed=self.seed, n_samples=self.n, n_feats=self.dim, 
                                      k = self.k, outliers=np.floor(self.n*self.noise), compactness_factor=overlap, scale=None)

        data, labels = gen.generate_data()
        del gen

        return data, labels
    
    def correlated(self, corr):
        """
        generate clusters with correlation between features

        :param corr: absolute value of correlation, between 0, 1

        :return data: 2d numpy array, each row is a data point
        :return labels: 1d numpy array, contains labels of each data point representing the cluster memberhsip
        """

        if corr < 0 or corr > 1:
            raise Exception('Invalid input value for "corr"! Values must be between 0 and 1.') 

        gen = mdcgen.ClusterGenerator(seed=self.seed, n_samples=self.n, n_feats=self.dim, 
                                      k = self.k, outliers=np.floor(self.n*self.noise), corr=corr)

        data, labels = gen.generate_data()
        del gen

        return data, labels
    
    def use_distribution(self, distribution):
        """
        change the distribution used to generate clusters

        :param distribution: a string
        """
        if not distribution in valid_distributions:
            raise Exception(f'distribution must be one of: {list(valid_distributions)}')
        
        if distribution == 'logistic':
            compactness = 0.05
        elif distribution == 'gaussian' or distribution == 'gap' or distribution == 'normal':
            compactness = 0.07
        elif distribution == 'gamma':
            compactness = 0.05
        else:
            compactness = 0.1
        gen = mdcgen.ClusterGenerator(seed=self.seed, n_samples=self.n, n_feats=self.dim, 
                                      k = self.k, outliers=np.floor(self.n*self.noise), distributions=distribution, compactness_factor=compactness)

        data, labels = gen.generate_data()
        del gen

        return data, labels
    

    def use_shape(self, shape, separation=3):
        """
        generate clusters with shape not limited to spherical
        but only 2 dimensional data is supported
        each feature roughly in the range[-1, 11]
 
        :param shape: string, one of 'sphere', 'concave', 'ring', 'line', 'star'
            if given a list of such strings, each one corresponds to a cluster
        :param separation: separation between cluster centres, default 3
        
        :return data: 2d numpy array, each row is a data point
        :return labels: 1d numpy array, contains labels of each data point representing the cluster memberhsip
        """

        if self.dim != 2:
            raise Exception('Not supported for dimensions other than 2')
        
        np.random.seed(self.seed)

        # allocate cluster sizes
        mean_size = self.n / self.k
        cluster_sizes = np.floor(np.random.normal(mean_size, mean_size/10, self.k)).astype(int)
        total_num = np.sum(cluster_sizes)
        # make sure the total number of data points is n
        if total_num > self.n:
            surplus = total_num - self.n
            cluster_sizes[np.argmax(cluster_sizes)] -= surplus
        else:
            lack = self.n - total_num
            cluster_sizes[np.argmin(cluster_sizes)] += lack

        # allocate cluster centres
        centre_x = np.zeros(self.k)
        centre_y = np.zeros(self.k)
        index = 0
        while index < self.k:
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
            # check enough separation between other cluster centres
            flag = True
            if index > 0:
                for i in range(index):
                    dist = np.sqrt((centre_x[i] - x) ** 2 + (centre_y[i] - y) ** 2)
                    if dist < separation:
                        flag = False
                        break
            
            if flag:
                centre_x[index] = x
                centre_y[index] = y
                index += 1

        if isinstance(shape, str):
            shape = [shape] * self.k
        if len(shape) != self.k:
            raise Exception('a list is given for shape, it must be as the same length of k(number of clusters)')

        data = np.zeros((self.n, 2))
        labels = np.zeros(self.n)
        start = 0
        for cluster in range(self.k):
            num = cluster_sizes[cluster]
            end = start + num
            labels[start:end] = cluster + 1

            if shape[cluster] == 'sphere':
                mean = [centre_x[cluster], centre_y[cluster]]
                cov = np.eye(2) / 4
                data[start:end, :] = np.random.multivariate_normal(mean, cov, num)
            elif shape[cluster] == 'line':
                angle = np.random.uniform(0, np.pi)
                line_len = np.random.uniform(1, 2) # length of line

                start_x = min(centre_x[cluster]-line_len*np.cos(angle), centre_x[cluster]+line_len*np.cos(angle))
                end_x = max(centre_x[cluster]-line_len*np.cos(angle), centre_x[cluster]+line_len*np.cos(angle))
                x = np.linspace(start_x, end_x, num=num)
                deviation_x = np.random.normal(0, 0.1, num)
                data[start:end, 0] = x + deviation_x

                start_y = min(centre_y[cluster]-line_len*np.sin(angle), centre_y[cluster]+line_len*np.sin(angle))
                end_y = max(centre_y[cluster]-line_len*np.sin(angle), centre_y[cluster]+line_len*np.sin(angle))
                y = np.linspace(start_y, end_y, num=num)
                deviation_y = np.random.normal(0, 0.1, num)
                data[start:end, 1] = y + deviation_y
            elif shape[cluster] == 'ring':
                x_c = centre_x[cluster]
                y_c = centre_y[cluster]
                ring_thickness = 0.6
                for i in np.arange(start, end):
                    angle = np.random.uniform(0, 2*np.pi)
                    r = np.random.uniform(1-ring_thickness/2, 1+ring_thickness/2)
                    data[i, 0] = x_c + r * np.cos(angle)
                    data[i, 1] = y_c + r * np.sin(angle)
            elif shape[cluster] == 'concave':
                crest_angle = np.random.uniform(0, 2*np.pi)
                # crest range and concaveness can be adjusted
                # crest range specifies how wide the crest is
                # concaveness must be EVEN number, higher value gives a more concave crest
                crest_range = 0.4*np.pi
                concaveness = 4
                coef = (0.7*np.pi / crest_range) ** concaveness
                radius = coef * (crest_range/2) ** concaveness
                x_c = centre_x[cluster]
                y_c = centre_y[cluster]
                for i in np.arange(start, end):
                    angle = np.random.uniform(0, 2*np.pi)
                    diff = coef*(angle - crest_angle) ** concaveness
                    r = np.random.uniform(0, min(diff, radius))
                    data[i, 0] = x_c + r * np.cos(angle)
                    data[i, 1] = y_c + r * np.sin(angle)
            elif shape[cluster] == 'star': # also called rose
                x_c = centre_x[cluster]
                y_c = centre_y[cluster]
                star_len = np.random.uniform(0.6, 2.3)
                k = np.random.choice(np.array([2, 3, 5, 7])) # number of pedals 
                for i in np.arange(start, end):
                    angle = np.random.uniform(0, 2*np.pi)
                    r = np.random.uniform(0, star_len*np.cos(k*angle))
                    data[i, 0] = x_c + r * np.cos(angle)
                    data[i, 1] = y_c + r * np.sin(angle)
            else:
                raise Exception('shape must be one of sphere, concave, ring, line, star')

            start = end

        return shuffle(data, labels)