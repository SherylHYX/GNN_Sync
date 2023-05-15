# standard libaries
import os
import random
import pickle as pk

# third-party libraries
import scipy.sparse as sp
import numpy as np
import numpy.random as rnd
import networkx as nx
from torch_geometric_signed_directed.data import DirectedData

# internel
from utils import Outliers_Model, preprocess_cycles, uscities_preprocess
from angular_baseline import ksync_spectral_baseline

def to_dataset_no_split(A, label, graph_labels, save_path, load_only=False, features=None, k=1):
    Ind_i, Ind_j, Ind_k = preprocess_cycles(A)
    if features is None:
        features = ksync_spectral_baseline(A, k, True)

    data = DirectedData(x=features, y=label,A=sp.csr_matrix(A), graph_labels=graph_labels, 
    Ind_i=Ind_i, Ind_j=Ind_j, Ind_k=Ind_k)
    if not load_only:
        if os.path.isdir(os.path.dirname(save_path)) == False:
            try:
                os.makedirs(os.path.dirname(save_path))
            except FileExistsError:
                print('Folder exists for best {}!'.format(os.path.dirname(save_path)))
        pk.dump(data, open(save_path, 'wb'))
    return data


def load_data_from_memory(root, name=None):
    data = pk.load(open(root, 'rb'))
    if os.path.isdir(root) == False:
        try:
            os.makedirs(root)
        except FileExistsError:
            pass
    return [data]


def load_data(args, random_seed):
    rnd.seed(random_seed)
    random.seed(random_seed)
    label = None
    default_name_base =  'trials' + str(args.num_trials)
    if args.dataset[:3] in ['ERO', 'BAO', 'RGG']:
        default_name_base += 'seed' + str(random_seed)
    elif args.dataset[:8] == 'uscities':
        default_name_base += 'seed' + str(random_seed)
    save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/'+args.dataset+default_name_base+'.pk')
    if (not args.regenerate_data) and os.path.exists(save_path):
        print('Loading existing data!')
        data = load_data_from_memory(save_path, name=None)[0]
    else:
        print('Generating new data!')
        if args.dataset[:3] in ['ERO', 'BAO', 'RGG']:
            if args.dataset[:3] in ['ERO', 'BAO']:
                measurement_graph = args.dataset[:2]
            elif args.dataset[:4] in ['RGGO']:
                measurement_graph = args.dataset[:3]
            A, labels, graph_labels = Outliers_Model(n=args.N, p=args.p, eta=args.eta, style=args.outlier_style, 
            measurement_graph=measurement_graph, k=args.k)
            # check connectivity!
            G = nx.from_scipy_sparse_matrix(A)
            assert nx.is_connected(G), 'Network not connected!'
            data = to_dataset_no_split(A, labels, graph_labels, save_path=save_path,
                        load_only=args.load_only, k=args.k)
        elif args.dataset[:8] == 'uscities':
            data = np.load('../real_data/uscities.npy')
            k_num = 50
            num_nodes = 1097
            patch_indices = np.load('../real_data/us_patch_indices_k50_thres6.npy')
            if os.path.exists('../real_data/us_added_noise_x_k50_thres6_100eta'+str(int(100*args.eta))+'seed'+str(random_seed)+'.npy'):
                added_noise_x = np.load('../real_data/us_added_noise_x_k50_thres6_100eta'+str(int(100*args.eta))+'seed'+str(random_seed)+'.npy')
                added_noise_y = np.load('../real_data/us_added_noise_y_k50_thres6_100eta'+str(int(100*args.eta))+'seed'+str(random_seed)+'.npy')
            else:
                added_noise_x = np.random.normal(0, args.eta*data[:, 0].std(), (num_nodes, k_num))
                added_noise_y = np.random.normal(0, args.eta*data[:, 1].std(), (num_nodes, k_num))
                np.save('../real_data/us_added_noise_x_k50_thres6_100eta'+str(int(100*args.eta))+'seed'+str(random_seed)+'.npy', added_noise_x)
                np.save('../real_data/us_added_noise_y_k50_thres6_100eta'+str(int(100*args.eta))+'seed'+str(random_seed)+'.npy', added_noise_y)

            A, labels, graph_labels = uscities_preprocess(args.eta, added_noise_x, added_noise_y, patch_indices, args.outlier_style)
            np.save('../real_data/us_adj_obs_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed)+'.npy', A)
            A = sp.csr_matrix(A)
            np.save('../real_data/us_angles_gt_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed)+'.npy', labels)
            data = to_dataset_no_split(A, labels, graph_labels, save_path=save_path,
                        load_only=args.load_only, k=args.k)
    label = (data.y, data.graph_labels)

    return label, data.x, sp.csr_matrix(data.A), data.Ind_i, data.Ind_j, data.Ind_k
