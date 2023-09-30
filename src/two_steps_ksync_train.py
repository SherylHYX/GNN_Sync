import os
import time
from datetime import datetime
from itertools import permutations

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss

# internal files
from angular_baseline import ksync_spectral_baseline
from metrics import ksync_calculate_upsets, ksync_calculate_upsets_and_confidence, compErrorViaMatrices
from metrics import ksync_cycle_inconsistency_loss
from GNN_models import DIGRAC_Unroll_kSync, Projected_Gradient_Step
from param_parser import parameter_parser
from preprocess import load_data

MSE_loss = MSELoss()
GNN_variant_names = ['innerproduct']
NUM_GNN_VARIANTS = len(GNN_variant_names) # number of GNN variants for each architecture

upset_choices = ['upset', 'cycle_inconsistency']
NUM_UPSET_CHOICES = len(upset_choices)
args = parameter_parser()
SPECTRA_STEP_NUM_TO_ADD = args.spectral_step_num
args.spectral_step_num = 0 # need to set this to zero then later add SPECTRA_STEP_NUM_TO_ADD steps
GPM_layer = Projected_Gradient_Step(args.alpha, False)
assert args.k > 1
torch.manual_seed(args.seed)
device = args.device
if args.cuda:
    print("Using cuda")
    torch.cuda.manual_seed(args.seed)
compare_names_all = []
GNN_names = ['GNNSync']
for method_name in args.all_methods:
    compare_names_all.append(method_name)

def evaluation(logstr, k, score, A_torch, Ind_i, Ind_j, Ind_k, label_np, val_index, test_index, SavePred, save_path, split, identifier_str):
    MSE_full = np.zeros((3, 1)) # dim 0 is for test val all, dim 1 is for mse_avg
    MSE_full[:] = np.nan
    score = score.detach().cpu().numpy()
    score_flipped = (-score) % (2*np.pi)
    # to obtain the best permutation of the predictions
    all_perm = list(permutations(range(k)))
    best_perm = all_perm[0]
    best_MSE = 10000
    for perm in all_perm:
        curr_MSE = 0
        for l in range(k):
            label_np_l = label_np[:, l]
            MSE1 = compErrorViaMatrices(score[:, perm[l]], label_np_l)
            MSE2 = compErrorViaMatrices(score_flipped[:, perm[l]], label_np_l)
            curr_MSE += min(MSE1, MSE2)
        curr_MSE = curr_MSE/k
        if curr_MSE < best_MSE:
            best_MSE = curr_MSE
            best_perm = perm
    # permute the predictions
    score_new = score.copy()
    for l in range(k):
        label_np_l = label_np[:, l]
        MSE1 = compErrorViaMatrices(score[:, best_perm[l]], label_np_l)
        MSE2 = compErrorViaMatrices(score_flipped[:, best_perm[l]], label_np_l)
        if MSE1 > MSE2:
            score_new[:, l] = score_flipped[:, best_perm[l]]
        else:
            score_new[:, l] = score[:, best_perm[l]]
    score = score_new

    score_torch = torch.FloatTensor(score).to(A_torch)

    upset, ind_labels, confidence = ksync_calculate_upsets_and_confidence(A_torch, score_torch, k)
    new_mat = torch.sparse_coo_tensor(indices=A_torch.nonzero().T, values=confidence, size=A_torch.shape).to_dense()             
    new_mat = new_mat.multiply(A_torch)
    new_mat = new_mat/new_mat.sum()*A_torch.sum()
    cycle_inconsistency_val = ksync_cycle_inconsistency_loss(new_mat, Ind_i, Ind_j, Ind_k, ind_labels, k).detach().item()
    upset_full = [upset.detach().item(), cycle_inconsistency_val]
    if SavePred:
        np.save(save_path+identifier_str+'_scores'+str(split), score.detach().cpu().numpy())

    logstr += '\n From ' + identifier_str + ':,'
    if label_np is not None:
        # test
        mse_sum = 0
        for l in range(k):
            mse_sum += compErrorViaMatrices(score[test_index, l], label_np[test_index, l])
        mse_avg = mse_sum/k
        outstrtest = 'Test average MSE:, {:.3f},'.format(mse_avg)
        MSE_full[0] = [mse_avg]
        
        # val
        mse_sum = 0
        for l in range(k):
            mse_sum += compErrorViaMatrices(score[val_index, l], label_np[val_index, l])
        mse_avg = mse_sum/k
        outstrval = 'Validation average MSE:, {:.3f},'.format(mse_avg)
        MSE_full[1] = [mse_avg]
        
        
        # all
        mse_sum = 0
        for l in range(k):
            mse_sum += compErrorViaMatrices(score[:, l], label_np[:, l])
        mse_avg = mse_sum/k
        outstrall = 'All average MSE:, {:.3f},'.format(mse_avg)
        MSE_full[2] = [mse_avg]
        
    logstr += outstrtest + outstrval + outstrall
    logstr += 'upset:,{:.6f}, cycle inconsistency value:, {:.6f},'.format(upset.detach().item(), cycle_inconsistency_val)
    return logstr, upset_full, MSE_full


class Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args, random_seed, save_name_base):
        """
        Constructing the trainer instance.
        :param args: Arguments object.
        """
        self.args = args
        self.device = args.device
        self.random_seed = random_seed

        label, self.features, self.A, self.Ind_i, self.Ind_j, self.Ind_k = load_data(args, random_seed)
        self.label, self.graph_labels = label
        self.features = torch.FloatTensor(self.features).to(args.device)
        self.args.N = self.A.shape[0]
        self.A_torch = torch.FloatTensor(self.A.toarray()).to(device)
        
        self.nfeat = self.features.shape[1]
        if self.label is not None:
            self.label = torch.FloatTensor(self.label).to(args.device)
            self.label_np = self.label.cpu().numpy()
        else:
            self.label_np = None
            self.label = None
        

        # date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        

        save_name = save_name_base + 'Seed' + str(random_seed)

        dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name)

        all_subdirs = [os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name, d) for d in os.listdir(dir_name)]
        self.log_path = max(all_subdirs, key=os.path.getmtime)

        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')

        self.runs = self.args.num_trials


    def train(self, model_name):
        #################################
        # training and evaluation
        #################################
        if model_name not in GNN_names:
            MSE_full, upset_full = self.non_nn(model_name)
            MSE_full_latest = MSE_full.copy()
            upset_full_latest = upset_full.copy()
        else:
            # (the last two dimensions) rows: test, val, all; cols: variants of MSE
            MSE_full = np.zeros([NUM_GNN_VARIANTS, self.runs, 3, 1])
            MSE_full[:] = np.nan
            MSE_full_latest = MSE_full.copy()

            upset_full = np.zeros([NUM_GNN_VARIANTS, self.runs, NUM_UPSET_CHOICES])
            upset_full[:] = np.nan
            upset_full_latest = upset_full.copy()
            
            args = self.args
            edge_index = torch.LongTensor(self.A.nonzero()).to(self.args.device)
            edge_weights = torch.FloatTensor(self.A.data).to(self.args.device)
            
            for split in range(self.runs):
                if self.args.sync_baseline == 'spectral':
                    score = ksync_spectral_baseline(self.A, self.args.k)
                elif self.args.sync_baseline == 'row_norm_spectral':
                    score = ksync_spectral_baseline(self.A, self.args.k, True)
                else:
                    raise NameError('Please input the correct baseline model name from:\
                        spectral, row_norm_spectral instead of {}!'.format(self.args.sync_baseline))
                score_torch = torch.FloatTensor(score).to(self.args.device)

                # to modify the features based on the synchronization baseline output
                self.features = score_torch
                self.nfeat = self.features.shape[1]

                if model_name == 'GNNSync':
                    model = DIGRAC_Unroll_kSync(num_features=self.nfeat, dropout=self.args.dropout, hop=self.args.hop, 
                embedding_dim=self.args.hidden*2, k=self.args.k, spectral_step_num=self.args.spectral_step_num, alpha=self.args.alpha, 
                trainable_alpha=self.args.trainable_alpha).to(self.args.device)

                else:
                    raise NameError('Please input the correct model name from:\
                        spectral, row_norm_spectral, GNNSync, instead of {}!'.format(model_name))

                M = self.A_torch
                # make A skew-symmetric
                A_skewed = self.A_torch - self.A_torch.T
                # Map to angular embedding (only the nonzero entries):
                H = torch.exp(1j * A_skewed).multiply(A_skewed != 0)

                train_index = np.ones(args.N, dtype=bool)
                val_index = train_index
                test_index = train_index

                ####################
                # Testing
                ####################
                base_save_path = self.log_path + '/'+model_name
                logstr = ''
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_model'+str(split)+'.t7'))
                model.eval()
                score = model(edge_index, edge_weights, self.features, self.args.separate_graphs)
                if self.args.separate_graphs:
                    score_l = score[:, 0:1]
                    T = (score_l - score_l.T) % (2*np.pi)
                    diff_v1 = (edge_weights-T[edge_index[0], edge_index[1]]) % (2*np.pi)
                    diff_v2 = (T[edge_index[0], edge_index[1]]-edge_weights) % (2*np.pi)
                    diff = torch.minimum(diff_v1, diff_v2)
                    diff_all = diff.clone()[None, :]
                    for l in range(1, score.shape[1]):
                        score_l = score[:, l:l+1]
                        T = (score_l - score_l.T) % (2*np.pi)
                        diff_v1 = (edge_weights-T[edge_index[0], edge_index[1]]) % (2*np.pi)
                        diff_v2 = (T[edge_index[0], edge_index[1]]-edge_weights) % (2*np.pi)
                        diff_l = torch.minimum(diff_v1, diff_v2)
                        diff_all = torch.cat((diff_all, diff_l[None, :]), dim=0)
                    ind_labels = torch.argmin(diff_all, dim=0)

                    for l in range(0, self.args.k):
                        if (ind_labels==l).sum() > 0:
                            score_l = score[:, l:l+1]
                            H_mask = torch.sparse_coo_tensor(indices=self.A_torch.nonzero()[ind_labels==l].T, values=torch.ones(len(self.A_torch.nonzero()[ind_labels==l])).to(self.A_torch), size=self.A_torch.shape).to_dense()
                            H_mask = H_mask + H_mask.T
                            H_torch = H.multiply(H_mask > 0)
                            for _ in range(SPECTRA_STEP_NUM_TO_ADD):
                                score[:, l:l+1] = GPM_layer(H_torch, score[:, l:l+1])
                else:
                    for l in range(self.args.k):
                        for _ in range(SPECTRA_STEP_NUM_TO_ADD):
                            score[:, l:l+1] = GPM_layer(H, score[:, l:l+1])
                score_model = score

                if self.args.upset_coeff > 0:
                    val_loss_upset, _ = ksync_calculate_upsets(M, score_model, self.args.k) 
                    test_loss_upset, _ = ksync_calculate_upsets(M, score_model, self.args.k)  
                    all_loss_upset, _ = ksync_calculate_upsets(M, score_model, self.args.k)                
                else:
                    val_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset = torch.ones(1, requires_grad=True).to(device)


                val_loss = self.args.upset_coeff * val_loss_upset
                test_loss = self.args.upset_coeff * test_loss_upset
                all_loss = self.args.upset_coeff * all_loss_upset

                logstr += 'Final results for {}:,'.format(model_name)
                logstr += 'Best val upset loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())

                logstr, upset_full[0, split], MSE_full[0, split] = evaluation(logstr, self.args.k, score_model, self.A_torch, self.Ind_i, self.Ind_j, self.Ind_k, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, '_best')

                # latest
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_model_latest'+str(split)+'.t7'))
                model.eval()
                score = model(edge_index, edge_weights, self.features, self.args.separate_graphs)
                if self.args.separate_graphs:
                    score_l = score[:, 0:1]
                    T = (score_l - score_l.T) % (2*np.pi)
                    diff_v1 = (edge_weights-T[edge_index[0], edge_index[1]]) % (2*np.pi)
                    diff_v2 = (T[edge_index[0], edge_index[1]]-edge_weights) % (2*np.pi)
                    diff = torch.minimum(diff_v1, diff_v2)
                    diff_all = diff.clone()[None, :]
                    for l in range(1, score.shape[1]):
                        score_l = score[:, l:l+1]
                        T = (score_l - score_l.T) % (2*np.pi)
                        diff_v1 = (edge_weights-T[edge_index[0], edge_index[1]]) % (2*np.pi)
                        diff_v2 = (T[edge_index[0], edge_index[1]]-edge_weights) % (2*np.pi)
                        diff_l = torch.minimum(diff_v1, diff_v2)
                        diff_all = torch.cat((diff_all, diff_l[None, :]), dim=0)
                    ind_labels = torch.argmin(diff_all, dim=0)

                    for l in range(0, self.args.k):
                        if (ind_labels==l).sum() > 0:
                            score_l = score[:, l:l+1]
                            H_mask = torch.sparse_coo_tensor(indices=self.A_torch.nonzero()[ind_labels==l].T, values=torch.ones(len(self.A_torch.nonzero()[ind_labels==l])).to(self.A_torch), size=self.A_torch.shape).to_dense()
                            H_mask = H_mask + H_mask.T
                            H_torch = H.multiply(H_mask > 0)
                            for _ in range(SPECTRA_STEP_NUM_TO_ADD):
                                score[:, l:l+1] = GPM_layer(H_torch, score[:, l:l+1])
                else:
                    for l in range(self.args.k):
                        for _ in range(SPECTRA_STEP_NUM_TO_ADD):
                            score[:, l:l+1] = GPM_layer(H, score[:, l:l+1])
                score_model = score
                
                if self.args.upset_coeff > 0:
                    val_loss_upset, _ = ksync_calculate_upsets(M, score_model, self.args.k) 
                    test_loss_upset, _ = ksync_calculate_upsets(M, score_model, self.args.k)  
                    all_loss_upset, _ = ksync_calculate_upsets(M, score_model, self.args.k)                
                else:
                    val_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset = torch.ones(1, requires_grad=True).to(device)


                val_loss = self.args.upset_coeff * val_loss_upset
                test_loss = self.args.upset_coeff * test_loss_upset
                all_loss = self.args.upset_coeff * all_loss_upset

                logstr += 'Latest val upset loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())


                logstr, upset_full_latest[0, split], MSE_full_latest[0, split] = evaluation(logstr, self.args.k, score_model, self.A_torch, self.Ind_i, self.Ind_j, self.Ind_k, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, '_latest')
                print(logstr)


                torch.cuda.empty_cache()
        return MSE_full, MSE_full_latest, upset_full, upset_full_latest

    def non_nn(self, model_name):
        #################################
        # training and evaluation for non-NN methods
        #################################
        MSE_full = np.zeros([self.runs, 3, 1])
        MSE_full[:] = np.nan
        upset_full = np.zeros([self.runs, NUM_UPSET_CHOICES])
        upset_full[:] = np.nan
        
        for split in range(self.runs):
            val_index = np.ones(args.N, dtype=bool)
            test_index = val_index

            ####################
            # Testing
            ####################
            logstr = ''

            if model_name == 'spectral':
                score = ksync_spectral_baseline(self.A, self.args.k)
            elif model_name == 'row_norm_spectral':
                score = ksync_spectral_baseline(self.A, self.args.k, True)
            elif model_name == 'trivial': # trivial solution
                score = np.ones_like(self.label.cpu().numpy())
            else:
                raise NameError('Please input the correct model name from:\
                    spectral, row_norm_spectral, trivial, GNNSync, instead of {}!'.format(model_name))

            score_flipped = (-score) % (2*np.pi)
            label_np = self.label_np
            k = self.args.k
            # to obtain the best permutation of the predictions
            all_perm = list(permutations(range(k)))
            best_perm = all_perm[0]
            best_MSE = 10000
            for perm in all_perm:
                curr_MSE = 0
                for l in range(k):
                    label_np_l = label_np[:, l]
                    MSE1 = compErrorViaMatrices(score[:, perm[l]], label_np_l)
                    MSE2 = compErrorViaMatrices(score_flipped[:, perm[l]], label_np_l)
                    curr_MSE += min(MSE1, MSE2)
                curr_MSE = curr_MSE/k
                if curr_MSE < best_MSE:
                    best_MSE = curr_MSE
                    best_perm = perm
            # permute the predictions
            score_new = score.copy()
            for l in range(k):
                label_np_l = label_np[:, l]
                MSE1 = compErrorViaMatrices(score[:, best_perm[l]], label_np_l)
                MSE2 = compErrorViaMatrices(score_flipped[:, best_perm[l]], label_np_l)
                if MSE1 > MSE2:
                    score_new[:, l] = score_flipped[:, best_perm[l]]
                else:
                    score_new[:, l] = score[:, best_perm[l]]
            score = score_new

            score_torch = torch.FloatTensor(score).to(self.args.device)

            upset, ind_labels, confidence = ksync_calculate_upsets_and_confidence(self.A_torch, score_torch, self.args.k)
            new_mat = torch.sparse_coo_tensor(indices=self.A_torch.nonzero().T, values=confidence, size=self.A_torch.shape).to_dense()             
            new_mat = new_mat.multiply(self.A_torch)
            new_mat = new_mat/new_mat.sum()*self.A_torch.sum()
            cycle_inconsistency_val = ksync_cycle_inconsistency_loss(new_mat, self.Ind_i, self.Ind_j, self.Ind_k, ind_labels, self.args.k).detach().item()
            upset_full = [upset.detach().item(), cycle_inconsistency_val]

            logstr += 'upset:,{:.6f}, cycle inconsistency:,{:.6f},'.format(upset.detach().item(), cycle_inconsistency_val)

            if self.args.SavePred:
                np.save(self.log_path + '/'+model_name+
                        '_scores'+str(split), score)
            
            print('Final results for {}:'.format(model_name))
            
            if label_np is not None:
            # test
                mse_sum = 0
                for l in range(k):
                    mse_sum += compErrorViaMatrices(score[test_index, l], label_np[test_index, l])
                mse_avg = mse_sum/k
                outstrtest = 'Test average MSE:, {:.3f},'.format(mse_avg)
                MSE_full[split, 0] = [mse_avg]
                
                # val
                mse_sum = 0
                for l in range(k):
                    mse_sum += compErrorViaMatrices(score[val_index, l], label_np[val_index, l])
                mse_avg = mse_sum/k
                outstrval = 'Validation average MSE:, {:.3f},'.format(mse_avg)
                MSE_full[split, 1] = [mse_avg]
                
                
                # all
                mse_sum = 0
                for l in range(k):
                    mse_sum += compErrorViaMatrices(score[:, l], label_np[:, l])
                mse_avg = mse_sum/k
                outstrall = 'All average MSE:, {:.3f},'.format(mse_avg)
                MSE_full[split, 2] = [mse_avg]
                    
                logstr += outstrtest + outstrval + outstrall

            print(logstr)
        return MSE_full, upset_full


# train and grap results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/'+args.dataset)


MSE_res = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 1]) # the last dim is the number of MSE versions
MSE_res_latest = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 1])

final_upset = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), NUM_UPSET_CHOICES])
final_upset[:] = np.nan
final_upset_latest = final_upset.copy()

method_str = ''
for method_name in args.all_methods:
    method_str += method_name

default_name_base = ''
if 'GNNSync' in args.all_methods:
    assert args.upset_coeff + args.cycle_coeff > 0, 'No loss to be optimized!'
    default_name_base += 'dropout' + str(int(100*args.dropout))
    default_name_base += 'upset_coe' + str(int(100*args.upset_coeff)) + 'cycle_coe' + str(int(100*args.cycle_coeff))
    if args.cycle_coeff > 0 and args.reg_coeff > 0:
        default_name_base += 'reg_coe' + str(int(100*args.reg_coeff))
    if args.spectral_step_num > 0:
        default_name_base += 'spectral_step_num' + str(args.spectral_step_num) + 'alpha' + str(int(100*args.alpha))
        default_name_base += 'train_alpha' + str(args.trainable_alpha)
        if args.separate_graphs:
            default_name_base += 'separate_graphs'
    default_name_base += 'hid' + str(args.hidden) + 'lr' + str(int(1000*args.lr))
    default_name_base += 'use' + str(args.sync_baseline)
    if args.optimizer != 'Adam':
        default_name_base += args.optimizer
save_name_base = default_name_base

default_name_base +=  'trials' + str(args.num_trials)
save_name_base = default_name_base
if args.dataset[:3] in ['ERO', 'BAO', 'RGG'] and set(args.seeds) != set([10, 20, 30, 40, 50]):
    default_name_base += 'seeds' + '_'.join([str(value) for value in np.array(args.seeds).flatten()])
save_name = default_name_base


current_seed_ind = 0
for random_seed in args.seeds:
    current_ind = 0
    trainer = Trainer(args, random_seed, save_name_base)
    for method_name in args.all_methods:
        if method_name not in GNN_names:
            MSE_full, MSE_full_latest, upset_full, upset_full_latest = trainer.train(method_name)
            MSE_res[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = MSE_full
            MSE_res_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = MSE_full_latest
            final_upset[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = upset_full
            final_upset_latest[current_ind, current_seed_ind: current_seed_ind + args.num_trials] = upset_full_latest
            current_ind += 1
        else:
            MSE_full, MSE_full_latest, upset_full, upset_full_latest = trainer.train(method_name)
            MSE_res[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = MSE_full
            MSE_res_latest[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = MSE_full_latest
            final_upset[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = upset_full
            final_upset_latest[current_ind: current_ind+NUM_GNN_VARIANTS, current_seed_ind: current_seed_ind + args.num_trials] = upset_full_latest
            current_ind += NUM_GNN_VARIANTS
    current_seed_ind += args.num_trials

# save results to arrays
for save_dir_name in ['MSE', 'upset']:
    if os.path.isdir(os.path.join(dir_name,save_dir_name,method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name,method_str))
        except FileExistsError:
            print('Folder exists for best {}!'.format(save_dir_name))
    if os.path.isdir(os.path.join(dir_name,save_dir_name+'_latest',method_str)) == False:
        try:
            os.makedirs(os.path.join(dir_name,save_dir_name+'_latest',method_str))
        except FileExistsError:
            print('Folder exists for latest {}!'.format(save_dir_name))

np.save(os.path.join(dir_name,'MSE',method_str,'two_step_'+save_name), MSE_res)
np.save(os.path.join(dir_name,'MSE_latest',method_str,'two_step_'+save_name), MSE_res_latest)
np.save(os.path.join(dir_name,'upset',method_str,'two_step_'+save_name), final_upset)
np.save(os.path.join(dir_name,'upset_latest',method_str,'two_step_'+save_name), final_upset_latest)
