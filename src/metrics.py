import torch
import numpy as np
from texttable import Texttable
import latextable

def calculate_upsets(A: torch.FloatTensor,
                     score: torch.FloatTensor)-> torch.FloatTensor:
    r"""Calculate upsets from angles. 
    
    Args:
        A: (torch.FloatTensor) Adjacency matrix.
        score: (torch.FloatTensor) Angles, with shape (num_nodes, 1).
        
    :rtype: 
        upset: (torch.FloatTensor) Portion of upsets, take average so that the value is bounded and normalized.
        """

    indices = (A != 0)
    score = score.reshape(score.shape[0], 1)
    T = (score - score.T) % (2*np.pi)
    diff1 = (A-T)[indices] % (2*np.pi)
    diff2 = (T-A)[indices] % (2*np.pi)
    diff = torch.minimum(diff1, diff2)
    upset = torch.mean(diff)
    return upset

def calculate_upsets_and_confidence(A: torch.FloatTensor,
                     score: torch.FloatTensor)-> torch.FloatTensor:
    r"""Calculate upsets from angles and confidence of edges. 
    
    Args:
        A: (torch.FloatTensor) Adjacency matrix.
        score: (torch.FloatTensor) Angles, with shape (num_nodes, 1).
        
    :rtype: 
        upset: (torch.FloatTensor) Portion of upsets, take average so that the value is bounded and normalized.
        confidence: (torch.FloatTensor) Confidence levels of the edges based on upset values.
        """

    indices = (A != 0)
    score = score.reshape(score.shape[0], 1)
    T = (score - score.T) % (2*np.pi)
    diff1 = (A-T)[indices] % (2*np.pi)
    diff2 = (T-A)[indices] % (2*np.pi)
    diff = torch.minimum(diff1, diff2)
    upset = torch.mean(diff)
    confidence = 1/(1 + diff)
    return upset, confidence


def compErrorViaMatrices(x, sol):
    '''
    Compute the MSE error (modulo the global shift) between the
    recovered solution and the ground  gruth
    (for general SO(d))
    '''
    ROT = mapToMatrices(x)
    EST_ROT = mapToMatrices(sol)
    MSE = compute_MSE_ROT(ROT, EST_ROT)

    return MSE


def mapToMatrices(x):
    cal=np.cos(x)
    sal=np.sin(x)

    n = len(x)
    ROT = np.zeros((n, 2, 2))
    for i in range(n):
        ROT[i] = np.array([[cal[i], -sal[i]],
                [sal[i], cal[i]]])
    return ROT


def compute_MSE_ROT(ROT, EST_ROT):
    d = ROT[0].shape[0]

    N = ROT.shape[0]

    Q = np.zeros((d, d))
    for i in range(N):
        Q = Q + np.matmul(ROT[i].T, EST_ROT[i])
    Q /= N
    _, S, _ = np.linalg.svd(Q)
    MSE = 4  - 2 * S.sum()

    return MSE

def cycle_inconsistency_loss(A_torch: torch.FloatTensor, Ind_i: np.array, Ind_j: np.array, Ind_k: np.array, reg_coeff: float=0):
    A_skewed_torch = (A_torch - A_torch.T) % (2*np.pi)
    v1 = A_skewed_torch[Ind_i, Ind_j] + A_skewed_torch[Ind_j, Ind_k] + A_skewed_torch[Ind_k, Ind_i]
    v2 = (-v1) % (2*np.pi)
    v1 = v1 % (2*np.pi)
    v = torch.minimum(v1, v2)
    return torch.mean(v) - reg_coeff * (A_skewed_torch[A_skewed_torch>0].std()) ** 2

def ksync_cycle_inconsistency_loss(A_torch: torch.FloatTensor, Ind_i: np.array, Ind_j: np.array, Ind_k: np.array, ind_labels: torch.LongTensor, k: int=2, reg_coeff: float=0):
    v_sum = 0
    var_sum = 0
    for l in range(k):
        if (ind_labels==l).sum():
            A_mask = torch.sparse_coo_tensor(indices=A_torch.nonzero()[ind_labels==l].T, values=torch.ones(len(A_torch.nonzero()[ind_labels==l])).to(A_torch), size=A_torch.shape).to_dense()
            A_mask = A_mask + A_mask.T
            A_skewed_torch = (A_torch - A_torch.T) % (2*np.pi)
            A_skewed_torch[A_mask==0] = 0
            v1 = (A_skewed_torch[Ind_i, Ind_j] + A_skewed_torch[Ind_j, Ind_k] + A_skewed_torch[Ind_k, Ind_i]).multiply(A_skewed_torch[Ind_i, Ind_j]>0).multiply(A_skewed_torch[Ind_j, Ind_k]>0).multiply(A_skewed_torch[Ind_k, Ind_i]>0)
            v2 = (-v1) % (2*np.pi)
            v1 = v1 % (2*np.pi)
            v_sum += torch.minimum(v1, v2)
            var_sum += (A_skewed_torch[A_skewed_torch>0].std()) ** 2
    v = v_sum/k
    var_avg = var_sum/k
    return torch.mean(v) - reg_coeff * var_avg

def ksync_calculate_upsets(A: torch.FloatTensor,
                     score: torch.FloatTensor, k: int)-> torch.FloatTensor:
    r"""Calculate upsets from angles. 
    
    Args:
        A: (torch.FloatTensor) Adjacency matrix.
        score: (torch.FloatTensor) Angles, with shape (num_nodes, 1).
        k: (int) Value k for k-synchronization.
        
    :rtype: 
        upset: (torch.FloatTensor) Portion of upsets, take average so that the value is bounded and normalized.
        ind_labels: (torch.LongTensor) Indices labels.
        """

    indices = (A != 0)
    score_l = score[:, 0:1]
    T = (score_l - score_l.T) % (2*np.pi)
    diff_v1 = (A-T)[indices] % (2*np.pi)
    diff_v2 = (T-A)[indices] % (2*np.pi)
    diff = torch.minimum(diff_v1, diff_v2)
    diff_all = diff.clone()[None, :]
    for l in range(1, k):
        score_l = score[:, l:l+1]
        T = (score_l - score_l.T) % (2*np.pi)
        diff_v1 = (A-T)[indices] % (2*np.pi)
        diff_v2 = (T-A)[indices] % (2*np.pi)
        diff_l = torch.minimum(diff_v1, diff_v2)
        diff = torch.minimum(diff, diff_l)
        diff_all = torch.cat((diff_all, diff_l[None, :]), dim=0)
    upset = torch.mean(diff)
    ind_labels = torch.argmin(diff_all, dim=0)
    return upset, ind_labels

def ksync_calculate_upsets_and_confidence(A: torch.FloatTensor,
                     score: torch.FloatTensor, k: int)-> torch.FloatTensor:
    r"""Calculate upsets from angles. 
    
    Args:
        A: (torch.FloatTensor) Adjacency matrix.
        score: (torch.FloatTensor) Angles, with shape (num_nodes, 1).
        k: (int) Value k for k-synchronization.
        
    :rtype: 
        upset: (torch.FloatTensor) Portion of upsets, take average so that the value is bounded and normalized.
        ind_labels: (torch.LongTensor) Indices labels.
        """

    indices = (A != 0)
    score_l = score[:, 0:1]
    T = (score_l - score_l.T) % (2*np.pi)
    diff_v1 = (A-T)[indices] % (2*np.pi)
    diff_v2 = (T-A)[indices] % (2*np.pi)
    diff = torch.minimum(diff_v1, diff_v2)
    diff_all = diff.clone()[None, :]
    for l in range(1, k):
        score_l = score[:, l:l+1]
        T = (score_l - score_l.T) % (2*np.pi)
        diff_v1 = (A-T)[indices] % (2*np.pi)
        diff_v2 = (T-A)[indices] % (2*np.pi)
        diff_l = torch.minimum(diff_v1, diff_v2)
        diff = torch.minimum(diff, diff_l)
        diff_all = torch.cat((diff_all, diff_l[None, :]), dim=0)
    upset = torch.mean(diff)
    ind_labels = torch.argmin(diff_all, dim=0)
    confidence = 1/(1 + diff)
    return upset, ind_labels, confidence

def print_overall_performance_mean_std(title:str, results:np.array, compare_names_all:list,
                               dataset_names:list, print_latex:bool=True, print_std:bool=True):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (deemed better with larger values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(dataset_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = title+'Data/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = dataset_names
    std = np.chararray(
        [len(dataset_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),3))
    results_mean = np.transpose(np.round(np.nanmean(results,0),3))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.3f}'.format(results_mean[i, j])
            std[i, j] = '{:.3f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(dataset_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=title +
                                    " performance.", label="table:"+title) + "\n")