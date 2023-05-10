import math
from math import atan2

import numpy as np
import scipy.sparse as sp
import networkx as nx

def Outliers_Model(n: int, p: float, eta: float, style:str='gamma', measurement_graph:str='ER', k: int=1) -> sp.csr_matrix:
    """An Outliers model graph generator with different measurement graph choices.
    Args:
        n: (int) Number of nodes.
        p: (float) Sparsity value, edge probability, or the radis for RGG model.
        eta : (float) Noise level, between 0 and 1.
        style: (string, optional) How to generate ground-truth angles:
            'gamma': Gamma distribution with shape 0.5 and scale 1. (default)
            'multi_normal0': Multivariate normal distribution with identity covariance matrix.
            'multi_normal1': Multivariate normal distribution with covariance matrix defined in the paper.
            'block_normal6': Multivariate normal distribution with the block-diagonal covariance matrix defined in the paper.
        measurement_graph: (string, optional) What measurement graph to set, default is 'ER', can also have 'BA', 'RGG'.
        k: (int, optional) The value k for k-synchronization data as detailed in 
            https://arxiv.org/abs/2012.14932, default 1 (normal angular synchronization).

    Returns:
        R: (sp.csr_matrix) a sparse n by n matrix of pairwise comparisons.
        labels: (np.array) ground-truth angles.
        graph_labels: (scipy.sparse matrix) groun-truth graph correspondence. 
            0 means no signal (no edge_index recorded), 1 means noise, 2 means the first graph, 3 means the second graph etc.
    """
    if style == 'uniform':
        scores = 2 * np.pi * np.random.rand(n, k)
        R_noise = (np.random.rand(n, n) * 2 - 1)  * 2 * np.pi
    elif style == 'gamma':
        scores = 2 * np.pi * np.random.gamma(shape=0.5, scale=1, size=(n, k))
        R_noise = 2 * np.pi * (np.random.rand(n, n) * 4 - 2) # 0.95 percentile for gamma(0.5, 1) is about 1.9207
    elif style == 'gamma_diff':
        scores = 2 * np.pi * np.random.gamma(shape=0.5, scale=1, size=(n, 1))
        if k>1:
            for l in range(1, k):
                scores_new = 2 * np.pi * np.random.gamma(shape=0.5*(l+1), scale=1, size=(n, 1))
                scores = np.concatenate((scores, scores_new), axis=1)
        R_noise = 2 * np.pi * (np.random.rand(n, n) * 4 - 2) # 0.95 percentile for gamma(0.5, 1) is about 1.9207
    elif style[:12] == 'multi_normal':
        U = int(style[12:])
        mean = np.pi*np.ones(n)
        if U == 0: # iid Gaussian
            cov = np.identity(n)
        else:
            a_vec = np.random.standard_normal((n, 1))
            cov = a_vec @ a_vec.T
            for i in range(2, U+1):
                a_vec = np.random.standard_normal((n, 1))
                cov += a_vec @ a_vec.T/i/i
        scores = np.random.multivariate_normal(mean, cov, k).T
        R_noise = (np.random.rand(n, n) * 2 - 1)  * 2 * np.pi
    elif style[:12] == 'block_normal':
        num_blocks = int(style[12:])
        mean = np.pi*np.ones(n)
        block_size = int(n/num_blocks)
        start_ind = 0
        cov = np.zeros((n, n))
        for _ in range(num_blocks - 1):
            a_vec = np.random.standard_normal((block_size, 1))
            cov[start_ind:(start_ind+block_size), start_ind:(start_ind+block_size)] = a_vec @ a_vec.T
            start_ind += block_size
        a_vec = np.random.standard_normal((n - start_ind, 1))
        cov[start_ind:n, start_ind: n] = a_vec @ a_vec.T
        scores = np.random.multivariate_normal(mean, cov, k).T
        R_noise = (np.random.rand(n, n) * 2 - 1)  * 2 * np.pi

    scores = scores % (2*np.pi)
    R_noise = R_noise % (2*np.pi)
    labels = scores
    R_GT_list = []
    for l in range(k):
        R_GT = scores[:, l:l+1] - scores[:, l:l+1].transpose() # use broadcasting
        R_GT_list.append(R_GT)
    
    R_choice = np.random.rand(n, n)
    R = np.zeros((n, n))
    graph_labels = np.ones((n, n))
    R[:] = R_noise[:]
    for l in range(k, 0, -1):
        R[R_choice<(1-eta)*l/k] = R_GT_list[l-1][R_choice<(1-eta)*l/k]
        graph_labels[R_choice<(1-eta)*l/k] = l+1
    if measurement_graph == 'ER':
        G = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    elif measurement_graph == 'BA':
        m = math.ceil(n * p / 2)
        G = nx.generators.random_graphs.barabasi_albert_graph(n, m)  # undirected
    elif measurement_graph == 'RGG': # random geometric graph
        G = nx.random_geometric_graph(n, 2*p)
    A_G = nx.adjacency_matrix(G).toarray()
    R[A_G==0] = 0
    lower_ind = np.tril_indices(n)
    diag_ind = np.diag_indices(n)
    R[lower_ind] = -R.transpose()[lower_ind]
    graph_labels[lower_ind] = graph_labels.transpose()[lower_ind]
    R[diag_ind] = 0
    R[R<0] = 0
    R = R % (2 * np.pi)
    graph_labels[R==0] = 0
    return sp.csr_matrix(R), labels, sp.csr_matrix(graph_labels)

def get_rotation(X, Y):
    """
    A part of a port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        angle = get_rotation(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    Outputs
    ------------
    angle 
        an angle of rotation that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # transformation matrix
    if my < m:
        T = T[:my,:]
    angle = atan2(T[1, 0], T[1, 1])
   
    return angle % (2*np.pi)

def get_rotation_matrix(theta):
    
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def uscities_preprocess(eta: float, style:str='gamma') -> sp.csr_matrix:
    """Preprocess uscities data.
    Args:
        eta : (float) Noise level, between 0 and 1.
        style: (string, optional) How to generate ground-truth angles:
            'gamma': Gamma distribution with shape 0.5 and scale 1. (default)
            'multi_normal0': Multivariate normal distribution with identity covariance matrix.
            'multi_normal1': Multivariate normal distribution with covariance matrix defined in the paper.
            'block_normal6': Multivariate normal distribution with the block-diagonal covariance matrix defined in the paper.

    Returns:
        R: (np.array) a sparse n by n matrix of pairwise comparisons.
        angles_gt: (np.array) ground-truth angles.
        graph_labels: (scipy.sparse matrix) groun-truth graph correspondence. 
            0 means no signal (no edge_index recorded), 1 means noise, 2 means the first graph, 3 means the second graph etc.
    """
    data = np.load('../real_data/uscities.npy')
    n = data.shape[0]
    num_nodes = n
    patch_indices = np.load('../real_data/us_patch_indices_k50_thres6_100eta'+str(int(100*eta))+'.npy')
    added_noise_x = np.load('../real_data/us_added_noise_x_k50_thres6_100eta'+str(int(100*eta))+'.npy')
    added_noise_y = np.load('../real_data/us_added_noise_y_k50_thres6_100eta'+str(int(100*eta))+'.npy')
    if style == 'uniform':
        angles_gt = 2 * np.pi * np.random.rand(n, 1)
    elif style == 'gamma':
        angles_gt = 2 * np.pi * np.random.gamma(shape=0.5, scale=1, size=(n, 1))
    elif style == 'gamma_diff':
        angles_gt = 2 * np.pi * np.random.gamma(shape=0.5, scale=1, size=(n, 1))
    elif style[:12] == 'multi_normal':
        U = int(style[12:])
        mean = np.pi*np.ones(n)
        if U == 0: # iid Gaussian
            cov = np.identity(n)
        else:
            a_vec = np.random.standard_normal((n, 1))
            cov = a_vec @ a_vec.T
            for i in range(2, U+1):
                a_vec = np.random.standard_normal((n, 1))
                cov += a_vec @ a_vec.T/i/i
        angles_gt = np.random.multivariate_normal(mean, cov, 1).T
    elif style[:12] == 'block_normal':
        num_blocks = int(style[12:])
        mean = np.pi*np.ones(n)
        block_size = int(n/num_blocks)
        start_ind = 0
        cov = np.zeros((n, n))
        for _ in range(num_blocks - 1):
            a_vec = np.random.standard_normal((block_size, 1))
            cov[start_ind:(start_ind+block_size), start_ind:(start_ind+block_size)] = a_vec @ a_vec.T
            start_ind += block_size
        a_vec = np.random.standard_normal((n - start_ind, 1))
        cov[start_ind:n, start_ind: n] = a_vec @ a_vec.T
        angles_gt = np.random.multivariate_normal(mean, cov, 1).T

    num_threshold = 6
    adj_obs = np.zeros((num_nodes, num_nodes)) # the observed angular differences matrix
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            patch_i = patch_indices[i]
            patch_j = patch_indices[j]
            common_nodes = list(set(patch_i).intersection(set(patch_j)))
            if len(common_nodes) >= num_threshold: # threshold of performing a Procrustes alignment is num_threshold (initially 6) nodes in common
                noisy_patch_i = data[patch_indices[i]].copy()
                noisy_patch_i[:, 0] += added_noise_x[i]
                noisy_patch_i[:, 1] += added_noise_y[i]
                noisy_patch_j = data[patch_indices[j]].copy()
                noisy_patch_j[:, 0] += added_noise_x[j]
                noisy_patch_j[:, 1] += added_noise_y[j]
                idx_i_sorted = np.argsort(patch_i)
                common_nodes_pos = np.searchsorted(patch_i[idx_i_sorted], common_nodes)
                indices_i = idx_i_sorted[common_nodes_pos]
                idx_j_sorted = np.argsort(patch_j)
                common_nodes_pos = np.searchsorted(patch_j[idx_j_sorted], common_nodes)
                indices_j = idx_j_sorted[common_nodes_pos]
                rotated_common_i = np.dot(get_rotation_matrix(angles_gt[i]), noisy_patch_i[indices_i].T).T
                rotated_common_j = np.dot(get_rotation_matrix(angles_gt[j]), noisy_patch_j[indices_j].T).T
                adj_obs[i, j] = get_rotation(rotated_common_j, rotated_common_i)
    R = adj_obs
    graph_labels = np.ones((num_nodes, num_nodes))
    graph_labels[R==0]=0

    return R, angles_gt, sp.csr_matrix(graph_labels)


def preprocess_cycles(A):
    # make A skew-symmetric
    A1 = A - A.transpose()
    # Map to angular embedding (only the nonzero entries):
    A1 = sp.triu(A1)
    A1.data = A1.data % (2*np.pi)
    A_original = A1.copy()
    row, col = A1.nonzero()
    m = len(A_original.data)
        
    # start CEMP iterations as initialization   
    # Matrix of codegree:
    # CoDeg[i,j] = 0 if i and j are not connected, otherwise,
    # CoDeg[i,j] = # of vertices that are connected to both i and j
    H_binary = A_original.copy()
    H_binary.data = np.ones(m)
    H_binary = H_binary + H_binary.transpose()
    # '''
    CoDeg = (H_binary @ H_binary).multiply(H_binary)
    # mark all the pairs of two nodes that have paths of length one but not paths of length two with -1
    CoDeg[(H_binary>0).multiply(CoDeg==0)] = -1
    # the elements of cycles:
    # --has path of length 1 but not length 2: -1
    # --has path of length 1 and path of length 2: # of such triangles
    # --has no path of length 1 nor 2: 0

    # grab the nonzero elements
    CoDeg_upper = np.triu(CoDeg.toarray(), 1)
    CoDeg_vec = CoDeg_upper.flatten()
    CoDeg_vec = CoDeg_vec[CoDeg_vec != 0]

    # get all the indices in CoDeg that corresponds to traingle
    cycles_pos_ind = np.where(CoDeg_vec > 0)[0]
    cycles_vec_pos = CoDeg_vec[cycles_pos_ind].astype(int)
    cum_ind = np.insert(np.cumsum(cycles_vec_pos), 0, 0)

    # total number of 3-cycles (triangles)
    m_pos = len(cycles_pos_ind)
    m_cycle = cum_ind[-1]

    Ind_i = np.zeros(m_cycle).astype(int)
    Ind_j = np.zeros(m_cycle).astype(int)
    Ind_k = np.zeros(m_cycle).astype(int)
    
    for l in range(m_pos):
        IJ = cycles_pos_ind[l]
        i = row[IJ]
        j = col[IJ]
        Ind_i[cum_ind[l]:cum_ind[l+1]] = i
        Ind_j[cum_ind[l]:cum_ind[l+1]] = j
        Ind_k[cum_ind[l]:cum_ind[l+1]] = np.where(H_binary[:, i].toarray().flatten() * H_binary[:, j].toarray().flatten())[0]

    return Ind_i, Ind_j, Ind_k