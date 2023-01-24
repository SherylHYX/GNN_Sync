import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx

def spectral_baseline(A, row_normalization=False):
    r"""The angular synchronization model from the
    `Angular synchronization by eigenvectors and semidefinite programming" 
    <https://www.sciencedirect.com/science/article/pii/S1063520310000205>`_ paper.
    Args:
        * **A** (scipy.sparse matrix): The underlying connectivity matrix, not skew-symmetric.
        * **row_normalization** (bool): Whether to row-normalize. Default is False.
    """
    A_torch = torch.FloatTensor(A.toarray())
    # make A skew-symmetric
    A = A - A.transpose()
    # Map to angular embedding (only the nonzero entries):
    H = A.copy()
    H.data = np.exp(1j * A.data) # 1j is the imaginary unit
    if row_normalization:
        H = sp.csr_matrix(H)
        D = sp.diags(np.array(1/(abs(H).sum(axis=1))).flatten())
        D.data = np.nan_to_num(D.data)
        H = H.dot(D)
    _, vecs = sp.linalg.eigsh(H, k=1, which='LA')
    theta_hat = (np.angle(vecs) % (2*np.pi)).flatten()
    return theta_hat

def ksync_spectral_baseline(A, k=2, row_normalization=False):
    r"""The spectral baselines.

    Args:
        * **A** (scipy.sparse matrix): The underlying connectivity matrix, not skew-symmetric.
        * **k** (int): Value k for k-synchronization.
        * **row_normalization** (bool): Whether to row-normalize. Default is False.
    """
    # make A skew-symmetric
    A = A - A.transpose()
    # Map to angular embedding (only the nonzero entries):
    H = A.copy()
    H.data = np.exp(1j * A.data) # 1j is the imaginary unit
    if row_normalization:
        H = sp.csr_matrix(H)
        D = sp.diags(np.array(1/(abs(H).sum(axis=1))).flatten())
        D.data = np.nan_to_num(D.data)
        H = H.dot(D)
    _, vecs = sp.linalg.eigsh(H, k=k, which='LA')
    theta_hat = (np.angle(vecs) % (2*np.pi))
    return theta_hat

def generalized_power_method(A, num_iter=1000):
    r"""The phase synchronization model using generalized power method (GPM) from the
    `Nonconvex Phase Synchronization" 
    <https://epubs.siam.org/doi/abs/10.1137/16M105808X>`_ paper.
    Args:
        * **A** (scipy.sparse matrix): The underlying connectivity matrix, not skew-symmetric.
        * **num_iter** (int, optional): Number of iterations, default 1000.
    """
    # make A skew-symmetric
    A = A - A.transpose()
    # Map to angular embedding (only the nonzero entries):
    H = A.copy()
    H.data = np.exp(1j * A.data) # 1j is the imaginary unit
    _, x = sp.linalg.eigsh(H, k=1, which='LA') # initialize
    x = x.flatten()
    x = x/abs(x)
    x = np.nan_to_num(x)

    lamb, _ = sp.linalg.eigsh(H, k=1, which='SA')
    lamb = lamb[0]
    alpha = max(0, -lamb)
    C_tilde = H + alpha * sp.diags(np.ones(H.shape[0]))
    for _ in range(num_iter):
        C_tilde_times_x = C_tilde.dot(x)
        if np.real(np.conj(x).dot(C_tilde_times_x))/np.linalg.norm(C_tilde_times_x, ord=1) >= 1 - 1e-7:
            break
        C_abs = abs(C_tilde_times_x)
        indicator = C_abs != 0
        x = (1 - indicator) * x + indicator * C_tilde_times_x/C_abs
    theta_hat = (np.angle(x) % (2*np.pi)).flatten()
    return theta_hat

def TranSync(A, niter=100):
    # make A skew-symmetric
    A = A - A.transpose()
    # Map to angular embedding (only the nonzero entries):
    H = A.copy()
    H.data = np.exp(1j * A.data) # 1j is the imaginary unit
    A = sp.triu(A)
    A.data = A.data % (2*np.pi)
    A_original = A.copy()
    row, col = A.nonzero()
    
    _, x = sp.linalg.eigsh(H, k=1, which='LA') # initialize

    
    theta_irls = (np.angle(x) % (2*np.pi)).flatten()
    irls_stop = 1
    iter = 1
    while iter <= niter and irls_stop >1e-3:
        thetaij_irls = (theta_irls[row] - theta_irls[col]) % (2*np.pi)
        resid_vec = ((thetaij_irls - A_original.data) % (2*np.pi))/np.pi
        resid_vec = np.minimum(resid_vec, 2 - resid_vec) + 1e-4;
        weight_vec = 1./resid_vec;
        Weights = A_original.copy()
        Weights.data = weight_vec
        Weights = Weights + Weights.transpose()

        # row normalization for Weights matrix
        Weights = sp.csr_matrix(Weights)
        D = sp.diags(np.array(1/(abs(Weights).sum(axis=1))).flatten())
        D.data = np.nan_to_num(D.data)
        Weights = Weights.dot(D)
        aijW = H.copy()
        aijW.data = aijW.data * Weights.data
        _, x = sp.linalg.eigsh(aijW, k=1, which='LA')
        theta_irls_new = (np.angle(x) % (2*np.pi)).flatten()
        diff_vec = ((theta_irls_new - theta_irls) % (2*np.pi))/np.pi
        irls_stop = np.mean(np.minimum(diff_vec, 2 - diff_vec))

        theta_irls =  theta_irls_new
        iter = iter + 1
                    
    return theta_irls

def CEMP(A, post_method='GCW', beta_init=1, beta_max=40, rate=1.2):
    # make A skew-symmetric
    A = A - A.transpose()
    # Map to angular embedding (only the nonzero entries):
    H = A.copy()
    H.data = np.exp(1j * A.data) # 1j is the imaginary unit
    A = sp.triu(A)
    A.data = A.data % (2*np.pi)
    A_original = A.copy()
    row, col = A.nonzero()
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

    Ind_ij = np.zeros(m_cycle).astype(int)
    Ind_jk = np.zeros(m_cycle).astype(int)
    Ind_ki = np.zeros(m_cycle).astype(int)



    IndMat = A_original.copy()
    IndMat.data = np.arange(m)
    IndMat = IndMat + IndMat.transpose()

    thetaijMat = A_original.copy()
    thetaji = (-A_original.data) % (2*np.pi)
    thetajiMat = A_original.copy()
    thetajiMat.data = thetaji
    thetaijMat = thetaijMat + thetajiMat

    
    jkvec = np.zeros(m_cycle)
    kivec = np.zeros(m_cycle)
    
    for l in range(m_pos):
        IJ = cycles_pos_ind[l]
        i = row[IJ]
        j = col[IJ]
        CoInd_ij= np.where(H_binary[:, i].toarray().flatten() * H_binary[:, j].toarray().flatten())[0]
        Ind_ij[cum_ind[l]:cum_ind[l+1]] =  IJ
        Ind_jk[cum_ind[l]:cum_ind[l+1]] =  (IndMat[j,CoInd_ij].toarray()).flatten()
        Ind_ki[cum_ind[l]:cum_ind[l+1]] =  (IndMat[CoInd_ij,i].toarray()).flatten()
        jkvec[cum_ind[l]:cum_ind[l+1]] =  (thetaijMat[j,CoInd_ij].toarray()).flatten()
        kivec[cum_ind[l]:cum_ind[l+1]] =  (thetaijMat[CoInd_ij,i].toarray()).flatten()

    
    ijvec = A_original.data[Ind_ij]
    


    # compute cycle-inconsistencies
    
    theta_cycle = ((ijvec + jkvec +kivec +6*np.pi) % 2*np.pi)/np.pi
    S0_long = np.minimum(theta_cycle, 2-theta_cycle)
    S0_vec = np.ones(m)

    Weight_vec = np.ones(m_cycle)
    S0_weight = S0_long * Weight_vec

    for l in range(m_pos):
        IJ = cycles_pos_ind[l]
        S0_vec[IJ] = np.sum(S0_weight[cum_ind[l]:cum_ind[l+1]])/np.sum(Weight_vec[cum_ind[l]:cum_ind[l+1]])


    # Initialization completed!

    
    # Reweighting Procedure Started ...
    SVec = S0_vec
    iter = 1
    beta = beta_init
    while beta <= beta_max:
        Sjk = SVec[Ind_jk]
        Ski = SVec[Ind_ki]
        S_sum = Ski + Sjk
        
        Weight_vec = np.exp(-beta*S_sum)
        S0_weight = S0_long * Weight_vec

        for l in range(m_pos):
            IJ = cycles_pos_ind[l]
            SVec[IJ] = np.sum(S0_weight[cum_ind[l]:cum_ind[l+1]])/np.sum(Weight_vec[cum_ind[l]:cum_ind[l+1]])


        beta = beta * rate
        iter = iter + 1

    # CEMP stage Completed!
    if post_method == 'GCW':
        SMat = A_original.copy()
        SMat.data = SVec
        SMat = SMat + SMat.transpose()

        tmpMat = SMat.copy()
        tmpMat.data = np.exp(-beta/rate*SMat.data)
        Weights = tmpMat.multiply(H_binary)
        # row normalization on Weights
        Weights = sp.csr_matrix(Weights)
        D = sp.diags(np.array(1/(abs(Weights).sum(axis=1))).flatten())
        D.data = np.nan_to_num(D.data)
        Weights = Weights.dot(D)
        aijW = H.copy()
        aijW.data = aijW.data * Weights.data
        _, x = sp.linalg.eigsh(aijW, k=1, which='LA')
        theta_est = (np.angle(x) % (2*np.pi)).flatten()
    elif post_method == 'MST':
        # Building minimum spanning tree ...
        SMat = A_original.copy()
        SMat.data = SVec + 1
        G = nx.from_scipy_sparse_matrix(SMat)
        if nx.is_connected(G):
            MST = nx.minimum_spanning_tree(G)
            A_G = nx.adjacency_matrix(MST)
            AdjTree = (A_G + A_G.transpose()).astype(bool)
            
            # compute thetai by multiplying thetaij along the spanning tree
            thetaij = A.data
            n = A_original.shape[0]
            rootnodes = [0]
            added = np.zeros(n)
            theta_est = np.zeros(n)
            theta_est[rootnodes] = 0
            added[rootnodes] = 1
            IndMat_sign = np.zeros((n,n))
            for l in range(m):
                i, j = row[l], col[l]
                IndMat_sign[i, j] = l # construct edge index matrix (for 2d-to-1d index conversion)
                IndMat_sign[j, i] = -l
                    
            while np.sum(added) < n:
                for node_root in rootnodes:
                    leaves = np.where((AdjTree[node_root].toarray().flatten())*(1-added)==1)[0]
                    for node_leaf in leaves:
                        edge_leaf = IndMat_sign[node_leaf,node_root]
                        if edge_leaf > 0:
                            theta_est[node_leaf] = (thetaij[int(abs(edge_leaf))] + theta_est[node_root]) % (2*np.pi)
                        else:
                            theta_est[node_leaf] = (-thetaij[int(abs(edge_leaf))] + theta_est[node_root]) % (2*np.pi)
                        added[node_leaf] = 1
                    rootnodes.extend(leaves)       
        else:
            theta_est = np.zeros(A_original.shape[0])
            theta_est[:] = np.nan

    return theta_est




