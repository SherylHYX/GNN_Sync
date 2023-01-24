import math

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
            'gamma_diff': Gamma distribution with shape 0.5*(l+1) for graph l and scale 1.
        measurement_graph: (string, optional) What measurement graph to set, default is 'ER', can also have 'BA', 'RGG'.
        k: (int, optional) The value k for k-synchronization data as detailed in 
            https://arxiv.org/abs/2012.14932, default 1 (normal angular synchronization).

    Returns:
        R: (sp.csr_matrix) a sparse n by n matrix of pairwise comparisons.
        labels: (np.array) ground-truth angles.
        graph_labels: (scipy.sparse matrix) groun-truth graph correspondence. 
            0 means no signal (no edge_index recorded), 1 means noise, 2 means the first graph, 3 means the second graph etc.
    """
    if style == 'gamma':
        scores = 2 * np.pi * np.random.gamma(shape=0.5, scale=1, size=(n, k))
        R_noise = 2 * np.pi * (np.random.rand(n, n) * 4 - 2) # 0.95 percentile for gamma(0.5, 1) is about 1.9207
    elif style == 'gamma_diff':
        scores = 2 * np.pi * np.random.gamma(shape=0.5, scale=1, size=(n, 1))
        if k>1:
            for l in range(1, k):
                scores_new = 2 * np.pi * np.random.gamma(shape=0.5*(l+1), scale=1, size=(n, 1))
                scores = np.concatenate((scores, scores_new), axis=1)
        R_noise = 2 * np.pi * (np.random.rand(n, n) * 4 - 2) # 0.95 percentile for gamma(0.5, 1) is about 1.9207

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