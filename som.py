from numpy import power, einsum, linalg, cov, linspace, random, sqrt, dot, zeros, subtract, nditer, unravel_index, arange, outer, exp
from math import pi
import scipy
from sys import stdout
from time import time
import numpy as np

def _incremental_index_verbose(m):
    
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    for i in range(m):
        yield i
        it_per_sec = (time() - beginning) / (i+1)
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {it_per_sec:4.5f} it/s'.format(it_per_sec=it_per_sec)
        stdout.write(progress)
        

class SOM:
    
    def __init__(self, dim_x, dim_y, n_var, learning_rate, sigma, f_neighborhood='gaussian'):
        
        # x-axis size of map
        self.dim_x = dim_x
        # y-axis size of map
        self.dim_y = dim_y
        # number of variables
        self.n_var = n_var
        # learning rate
        self.learning_rate = learning_rate
        # sigma
        self.sigma = sigma
        # incremental iteration
        self.inc_iter = 0
        
        self.dist_map = zeros((dim_x, dim_y))
        
        if f_neighborhood == 'gaussian':
            self.neighborhood = self.f_gaussian
        elif f_neighborhood == 'mexican':
            self.neighborhood = self.f_mexican
        elif f_neighborhood == 'bubble':
            self.neighborhood = self.f_bubble
        else:
            self.neighborhood = self.f_triangle
        
    def f_gaussian(self, x, y, sigma):
        d = 2*pi*sigma*sigma
        ax = exp(-power(arange(self.dim_x)-x, 2)/d)
        ay = exp(-power(arange(self.dim_y)-y, 2)/d)
        return outer(ax, ay)  

    def f_mexican(self, x, y, sigma):
        xx, yy = meshgrid(arange(self.dim_x), arange(self.dim_y))
        p = power(xx-x, 2) + power(yy-y, 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)
    
    def f_bubble(self, x, y, sigma):
        ax = (arange(self.dim_x) > (x-sigma)) & (arange(self.dim_x) < (x+sigma))
        ay = (arange(self.dim_y) > (y-sigma)) & (arange(self.dim_y) < (y+sigma))
        return outer(ax, ay)*1.

    def f_triangle(self, x, y, sigma):
        triangle_x = (-abs(x - arange(self.dim_x))) + sigma
        triangle_y = (-abs(y - arange(self.dim_y))) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)
    
    # initialize weights
    def random_init_w(self):
        weights = random.rand(self.dim_x, self.dim_y, self.n_var)
        norm = np.apply_along_axis(lambda x: sqrt(dot(x, x.T)), 2, weights)
        self._weights = (1/norm[:, :, np.newaxis])*weights
        
    def pca_init_w(self, data):
        pc_length, pc = np.linalg.eig(np.cov(np.transpose(data)))
        pc_order = np.argsort(pc_length)
        self._weights = np.zeros((self.dim_x, self.dim_y, self.n_var))
        
        for i, c1 in enumerate(linspace(-1, 1, self.dim_x)):
            for j, c2 in enumerate(linspace(-1, 1, self.dim_y)):
                self._weights[i, j] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]]
     

    def execute_train(self, data, n_iter_data, init_method='pca'):
        
        if init_method == 'pca':
            self.pca_init_w(data)
        else:
            self.random_init_w()
        
        n_iter = data.shape[0]*n_iter_data
        
        arr_data_indx = arange(data.shape[0])
        
        self.list_dist_convg = []
        self.list_quantization_error = []
        self.list_topgraphic_error = []
        
        n_run = _incremental_index_verbose(n_iter)
        
        for i_ter in n_run:
            
            if (i_ter % data.shape[0]) == 0:
                np.random.shuffle(arr_data_indx)
            
            i_dx = i_ter % data.shape[0]
            i_dx = arr_data_indx[i_dx]
            self.update(data[i_dx], self.winner(data[i_dx]), self.inc_iter, n_iter)
            self.inc_iter += 1
            
            
            if (i_ter % data.shape[0]) == 0:
                self.list_dist_convg.append(self.dist_convg(data, self._weights, 0.05))
                self.list_quantization_error.append(self.quantization_error(data))
                self.list_topgraphic_error.append(self.topgraphic_error(data, self.inc_iter, n_iter))
            
      
    # build up the distance matrix between x and weights
    def _activate(self, x):
        s = subtract(x, self._weights)  # x - w
        it = nditer(self.dist_map, flags=['multi_index'])
        
        while not it.finished:
            self.dist_map[it.multi_index] = sqrt(dot(s[it.multi_index], s[it.multi_index].T))
            it.iternext()
            
    # find out the coordinate of a winner for x
    def winner(self, x):
        self._activate(x)
        return unravel_index(self.dist_map.argmin(),self.dist_map.shape)
    
    # update step
    def update(self, x, win, i_iter, n_iter):
        # calculate the learning rate and sigma for this step
        eta = self.f_decay(self.learning_rate, i_iter, n_iter)
        sig = self.f_decay(self.sigma, i_iter, n_iter)
        # improves the performances
        g = self.neighborhood(win[0], win[1], sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)
        
    # decay function for convergence
    def f_decay(self, learning_rate, i_iter, n_iter):
        return learning_rate / (1+i_iter/(n_iter/2))
    
    # performance: topographic error
    def i_topograohic_error(self, x, eta, sig):
        
        self._activate(x)
        x = np.c_[np.unravel_index(np.argsort(self.dist_map.flatten())[:2], self.dist_map.shape)]
        
        g = self.neighborhood(x[0][0], x[0][1], sig)
        return g[x[1][0], x[1][1]]
    def topgraphic_error(self, data, i_iter, n_iter):
        
        eta = self.f_decay(self.learning_rate, i_iter, n_iter)
        sig = self.f_decay(self.sigma, i_iter, n_iter)
        
        return np.mean(np.apply_along_axis(self.i_topograohic_error, 1, data, eta=eta, sig=sig))
    
    # performance: quantization error
    def quantization_error(self, data):
        
        n_sample = data.shape[0]
        error = 0
        for x in data:
            x_dist = x-self._weights[self.winner(x)]
            error += sqrt(dot(x_dist, x_dist.T))
            
        return error/n_sample
    
    
    # performance: distribution convergence
    def F_test(self, X, Y):
        df1 = len(X) - 1
        df2 = len(Y) - 1
        F = np.var(X)/np.var(Y)
        p_value = scipy.stats.f.cdf(F, df1, df2)
        return p_value
    def t_test(self, X, Y):
        p_value = scipy.stats.ttest_ind(X, Y, equal_var=False)
        return p_value.pvalue
    def dist_convg(self, X, Y, significant_level):
        a = [(self.t_test(X[:, i], Y[:, :, i].flatten()), self.F_test(X[:, i], Y[:, :, i].flatten())) for i in range(X.shape[1])]
        t, f = tuple(zip(*a))
        
        return np.mean((np.array(t)>= significant_level) & (np.array(f)>= significant_level))
    
    def distance_map(self):
        
        um = zeros((self._weights.shape[0], self._weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if (ii >= 0 and ii < self._weights.shape[0] and
                            jj >= 0 and jj < self._weights.shape[1]):
                        w_1 = self._weights[ii, jj, :]
                        w_2 = self._weights[it.multi_index]
                        w = w_1 - w_2
                        um[it.multi_index] += sqrt(dot(w, w.T))
            it.iternext()
        um = um/um.max()
        return um
    
