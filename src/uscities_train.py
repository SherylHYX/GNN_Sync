import os
import time
import math
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# internal files
from angular_baseline import spectral_baseline, generalized_power_method
from angular_baseline import TranSync, CEMP, trimmed_averaging_synchronization
from metrics import calculate_upsets, compErrorViaMatrices, calculate_upsets_and_confidence, cycle_inconsistency_loss
from GNN_models import DIGRAC_Unroll_Sync
from param_parser import parameter_parser
from preprocess import load_data


def get_rotation_matrix(theta):
    
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])



MSE_loss = MSELoss()
GNN_variant_names = ['innerproduct']
NUM_GNN_VARIANTS = len(GNN_variant_names) # number of GNN variants for each architecture

upset_choices = ['upset', 'cycle_inconsistency']
NUM_UPSET_CHOICES = len(upset_choices)
args = parameter_parser()
args.dataset = 'uscities/100eta'+str(int(100*args.eta))+args.outlier_style
args.num_trials = 1
args.seeds = [10]
whole_map = np.load('../real_data/uscities.npy')
patch_indices = np.load('../real_data/us_patch_indices_k50_thres6_100eta'+str(int(100*args.eta))+'.npy')
added_noise_x = np.load('../real_data/us_added_noise_x_k50_thres6_100eta'+str(int(100*args.eta))+'.npy')
added_noise_y = np.load('../real_data/us_added_noise_y_k50_thres6_100eta'+str(int(100*args.eta))+'.npy')
torch.manual_seed(args.seed)
device = args.device
if args.cuda:
    print("Using cuda")
    torch.cuda.manual_seed(args.seed)
compare_names_all = []
GNN_names = ['GNNSync']
for method_name in args.all_methods:
    compare_names_all.append(method_name)

def evaluation(logstr, score, A_torch, Ind_i, Ind_j, Ind_k, label_np, val_index, test_index, SavePred, save_path, split, identifier_str):
    MSE_full = np.zeros((3, 1))
    MSE_full[:] = np.nan
    score_torch = score.clone()
    score = score.detach().cpu().numpy().flatten()
    score2 = (-score) % (2*np.pi)
    MSE1 = compErrorViaMatrices(score, label_np)
    MSE2 = compErrorViaMatrices(score2, label_np)
    if MSE1 > MSE2:
        score = score2
    upset, confidence = calculate_upsets_and_confidence(A_torch, score_torch)
    new_mat = torch.sparse_coo_tensor(indices=A_torch.nonzero().T, values=confidence, size=A_torch.shape).to_dense()             
    new_mat = new_mat.multiply(A_torch)
    new_mat = new_mat/new_mat.sum()*A_torch.sum()
    cycle_inconsistency_val = cycle_inconsistency_loss(new_mat, Ind_i, Ind_j, Ind_k).detach().item()
    upset_full = [upset.detach().item(), cycle_inconsistency_val]
    if SavePred:
        np.save(save_path+identifier_str+'_scores'+str(split), score.detach().cpu().numpy())

    logstr += '\n From ' + identifier_str + ':,'
    if label_np is not None:
        # test
        mse1 = compErrorViaMatrices(score[test_index], label_np[test_index])
        outstrtest = 'Test MSE:, {:.3f}, '.format(mse1)
        MSE_full[0] = [mse1]
        
        # val
        mse1 = compErrorViaMatrices(score[val_index], label_np[val_index])
        outstrval = 'Validation MSE:, {:.3f},'.format(mse1)
        MSE_full[1] = [mse1]
        
        
        # all
        mse1 = compErrorViaMatrices(score, label_np)
        outstrall = 'All MSE:, {:.3f},'.format(mse1)
        MSE_full[2] = [mse1]
        
    logstr += outstrtest + outstrval + outstrall
    logstr += 'upset:,{:.6f}, cycle inconsistency value:, {:.6f}, '.format(upset.detach().item(), cycle_inconsistency_val)
    if identifier_str == '_best':
        # plot
        angles_diff = (score-label_np) % (2*np.pi)
        score = (score - angles_diff.mean()) % (2*np.pi)
        np.save('../uscities_pred/GNNSync_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed), score)
        for i in range(A_torch.shape[0]):
            noisy_patch_i = whole_map[patch_indices[i]].copy()
            noisy_patch_i[:, 0] += added_noise_x[i]
            noisy_patch_i[:, 1] += added_noise_y[i]
            rotated_coordinates = np.dot(get_rotation_matrix(label_np[i]), noisy_patch_i.T).T
            rotated_coordinates = np.dot(get_rotation_matrix(-score[i]), rotated_coordinates.T).T
            plt.scatter(noisy_patch_i[:, 0], noisy_patch_i[:, 1], s=0.5, alpha=0.8, c='yellow')
            plt.scatter(rotated_coordinates[:, 0], rotated_coordinates[:, 1], s=0.5, alpha=0.8, c='blue')
        plt.scatter(whole_map[:, 0], whole_map[:, 1], s=1, c='red')
        plt.title('MSE={:.3f}'.format(MSE_full[2][0]))
        # plt.savefig('../uscities_plots/GNNSync_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed)+'.pdf',format='pdf')
        plt.savefig('../uscities_plots/GNNSync_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed)+'.png',format='png')
        plt.show()
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
        self.features = torch.FloatTensor(self.features).to(args.device)
        self.label, self.graph_labels = label
        self.args.N = self.A.shape[0]
        self.A_torch = torch.FloatTensor(self.A.toarray()).to(device)
        
        self.nfeat = self.features.shape[1]
        if self.label is not None:
            self.label = torch.FloatTensor(self.label).to(args.device)
            self.label_np = self.label.cpu().numpy().flatten()
        else:
            self.label_np = None

        date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        

        save_name = save_name_base + 'Seed' + str(random_seed)

        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name, date_time)

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
                    score = spectral_baseline(self.A)
                elif self.args.sync_baseline == 'row_norm_spectral':
                    score = spectral_baseline(self.A, True)
                elif self.args.sync_baseline == 'GPM':
                    score = generalized_power_method(self.A)
                elif self.args.sync_baseline == 'GPM':
                    score = generalized_power_method(self.A)
                elif self.args.sync_baseline == 'TranSync':
                    score = TranSync(self.A)
                elif self.args.sync_baseline == 'CEMP_GCW':
                    score = CEMP(self.A, post_method='GCW')
                elif self.args.sync_baseline == 'CEMP_MST':
                    score = CEMP(self.A, post_method='MST')
                elif self.args.sync_baseline == 'TAS':
                    score = trimmed_averaging_synchronization(self.A)
                else:
                    raise NameError('Please input the correct baseline model name from:\
                        spectral, row_norm_spectral, GPM, TranSync, CEMP_GCW, CEMP_MST, TAS instead of {}!'.format(self.args.sync_baseline))
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)

                upset1 = calculate_upsets(self.A_torch, score_torch)
                upset2 = calculate_upsets(torch.transpose(self.A_torch, 0, 1), score_torch)
                
                if upset1.detach().item() > upset2.detach().item():
                    score = -score
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)

                # to modify the features based on the synchronization baseline output
                self.features = score_torch
                self.nfeat = self.features.shape[1]

                if model_name == 'GNNSync':
                    model = DIGRAC_Unroll_Sync(num_features=self.nfeat, dropout=self.args.dropout, hop=self.args.hop, 
                embedding_dim=self.args.hidden*2, spectral_step_num=self.args.spectral_step_num, alpha=self.args.alpha, 
                trainable_alpha=self.args.trainable_alpha).to(self.args.device)
                else:
                    raise NameError('Please input the correct model name from:\
                        spectral, row_norm_spectral, GPM, TranSync, CEMP_GCW, CEMP_MST, GNNSync, instead of {}!'.format(model_name))

                if self.args.optimizer == 'Adam':
                    opt = optim.Adam(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                elif self.args.optimizer == 'SGD':
                    opt = optim.SGD(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                else:
                    raise NameError('Please input the correct optimizer name, Adam or SGD!')
                M = self.A_torch

                train_index = np.ones(args.N, dtype=bool)
                val_index = train_index
                test_index = train_index
                #################################
                # Train/Validation/Test
                #################################
                best_val_loss = 1000.0
                early_stopping = 0
                log_str_full = ''
                confidence = torch.ones(len(edge_weights)).to(self.args.device)
                
                for epoch in range(args.epochs):
                    start_time = time.time()
                    ####################
                    # Train
                    ####################

                    model.train()
                    score = model(edge_index, edge_weights, self.features)
                    
                    
                    train_loss_upset, confidence = calculate_upsets_and_confidence(M, score)  
                    new_mat = torch.sparse_coo_tensor(indices=self.A_torch.nonzero().T, values=confidence, size=self.A_torch.shape).to_dense()             
                    new_mat = new_mat.multiply(self.A_torch)
                    new_mat = new_mat/new_mat.sum()*self.A_torch.sum()
                    if self.args.cycle_coeff > 0:
                        train_loss_inconsistency = cycle_inconsistency_loss(new_mat, self.Ind_i, self.Ind_j, self.Ind_k, self.args.reg_coeff)
                    else:
                        train_loss_inconsistency = torch.ones(1, requires_grad=True).to(device)

                    train_loss = self.args.upset_coeff * train_loss_upset + self.args.cycle_coeff * train_loss_inconsistency
                    outstrtrain = 'Train loss:, {:.6f}, upset loss: {:6f}, cycle inconsistency loss, {:6f},'.format(train_loss.detach().item(),
                    train_loss_upset.detach().item(), train_loss_inconsistency.detach().item())
                    opt.zero_grad()

                    try:
                        train_loss.backward()
                    except RuntimeError:
                        log_str = '{} trial {} RuntimeError!'.format(model_name, split)
                        log_str_full += log_str + '\n'
                        print(log_str)
                        if not os.path.isfile(self.log_path + '/'+model_name+'_model'+str(split)+'.t7'):
                                torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model'+str(split)+'.t7')
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model_latest'+str(split)+'.t7')
                        break

                    opt.step()
                    ####################
                    # Validation
                    ####################
                    model.eval()
                    score = model(edge_index, edge_weights, self.features)

                    if self.args.upset_coeff > 0:
                        val_loss_upset = calculate_upsets(M, score_torch)               
                    else:
                        val_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    if self.args.cycle_coeff > 0:
                        val_loss_inconsistency = cycle_inconsistency_loss(new_mat, self.Ind_i, self.Ind_j, self.Ind_k, self.args.reg_coeff)
                    else:
                        val_loss_inconsistency = torch.ones(1, requires_grad=True).to(device)

                    
                    val_loss = self.args.upset_coeff * val_loss_upset + self.args.cycle_coeff * val_loss_inconsistency


                    outstrval = 'val loss:, {:.6f}, upset loss: {:6f}, cycle inconsistency loss, {:6f},'.format(val_loss.detach().item(),
                    val_loss_upset.detach().item(), val_loss_inconsistency.detach().item())

                    duration = "---, {:.4f}, seconds ---".format(
                        time.time() - start_time)
                    log_str = ("{}, / {} epoch,".format(epoch, args.epochs)) + \
                        outstrtrain+outstrval+duration
                    log_str_full += log_str + '\n'
                    print(log_str)
                    
                    ####################
                    # Save weights
                    ####################
                    save_perform = val_loss.detach().item()
                    if save_perform <= best_val_loss:
                        early_stopping = 0
                        best_val_loss = save_perform
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model'+str(split)+'.t7')
                    else:
                        early_stopping += 1
                    if early_stopping > args.early_stopping or epoch == (args.epochs-1):
                        torch.save(model.state_dict(), self.log_path +
                                '/'+model_name+'_model_latest'+str(split)+'.t7')
                        break

                status = 'w'
                if os.path.isfile(self.log_path + '/'+model_name+'_log'+str(split)+'.csv'):
                    status = 'a'
                with open(self.log_path + '/'+model_name+'_log'+str(split)+'.csv', status) as file:
                    file.write(log_str_full)
                    file.write('\n')
                    status = 'a'

                ####################
                # Testing
                ####################
                base_save_path = self.log_path + '/'+model_name
                logstr = ''
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_model'+str(split)+'.t7'))
                model.eval()
                score_model = model(edge_index, edge_weights, self.features)

                if self.args.upset_coeff > 0:
                    val_loss_upset = calculate_upsets(M, score_model) 
                    test_loss_upset = calculate_upsets(M, score_model)  
                    all_loss_upset = calculate_upsets(M, score_model)                
                else:
                    val_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset = torch.ones(1, requires_grad=True).to(device)


                val_loss = self.args.upset_coeff * val_loss_upset
                test_loss = self.args.upset_coeff * test_loss_upset
                all_loss = self.args.upset_coeff * all_loss_upset

                logstr += 'Final results for {}:,'.format(model_name)
                logstr += 'Best val upset loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())

                logstr, upset_full[0, split], MSE_full[0, split] = evaluation(logstr, score_model, self.A_torch, self.Ind_i, self.Ind_j, self.Ind_k, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, '_best')
                

                # latest
                model.load_state_dict(torch.load(
                    self.log_path + '/'+model_name+'_model_latest'+str(split)+'.t7'))
                model.eval()
                score_model = model(edge_index, edge_weights, self.features)
                
                if self.args.upset_coeff > 0:
                    val_loss_upset = calculate_upsets(M, score_model) 
                    test_loss_upset = calculate_upsets(M, score_model)  
                    all_loss_upset = calculate_upsets(M, score_model)                
                else:
                    val_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    test_loss_upset = torch.ones(1, requires_grad=True).to(device)
                    all_loss_upset = torch.ones(1, requires_grad=True).to(device)


                val_loss = self.args.upset_coeff * val_loss_upset
                test_loss = self.args.upset_coeff * test_loss_upset
                all_loss = self.args.upset_coeff * all_loss_upset

                logstr += 'Latest val upset loss: ,{:.3f}, test loss: ,{:.3f}, all loss: ,{:.3f},'.format(val_loss.detach().item(), test_loss.detach().item(), all_loss.detach().item())


                logstr, upset_full_latest[0, split], MSE_full_latest[0, split] = evaluation(logstr, score_model, self.A_torch, self.Ind_i, self.Ind_j, self.Ind_k, self.label_np, val_index, test_index, self.args.SavePred, \
                    base_save_path, split, '_latest')

                print(logstr)

                with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', status) as file:
                    file.write(logstr)
                    file.write('\n')

                torch.cuda.empty_cache()
        return MSE_full, MSE_full_latest, upset_full, upset_full_latest

    def non_nn(self, model_name):
        #################################
        # training and evaluation for non-NN methods
        #################################
        MSE_full = np.zeros([self.runs, 3, 1]) # 3 is all+val+test, 1 is one version of MSE
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
                score = spectral_baseline(self.A)
            elif model_name == 'row_norm_spectral':
                score = spectral_baseline(self.A, True)
            elif model_name == 'GPM':
                score = generalized_power_method(self.A)
            elif model_name == 'TranSync':
                try:
                    score = TranSync(self.A)
                except Exception:
                    score = np.zeros(self.A.shape[0])
                    score[:] = np.nan
            elif model_name == 'CEMP_GCW':
                score = CEMP(self.A, post_method='GCW')
            elif model_name == 'CEMP_MST':
                score = CEMP(self.A, post_method='MST')
            elif model_name == 'TAS':
                score = trimmed_averaging_synchronization(self.A)
            elif model_name == 'trivial': # trivial solution
                score = np.ones(self.A.shape[0])
            else:
                raise NameError('Please input the correct model name from:\
                    spectral, row_norm_spectral, GPM, TranSync, CEMP_GCW, CEMP_MST, TAS, GNNSync, trivial, instead of {}!'.format(model_name))
            if self.label is not None:
                score2 = (-score) % (2*np.pi)
                MSE1 = compErrorViaMatrices(score, self.label_np)
                MSE2 = compErrorViaMatrices(score2, self.label_np)
                if MSE1 > MSE2:
                    score = score2
            score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)

            upset, confidence = calculate_upsets_and_confidence(self.A_torch, score_torch)
            new_mat = torch.sparse_coo_tensor(indices=self.A_torch.nonzero().T, values=confidence, size=self.A_torch.shape).to_dense()             
            new_mat = new_mat.multiply(self.A_torch)
            new_mat = new_mat/new_mat.sum()*self.A_torch.sum()
            cycle_inconsistency_val = cycle_inconsistency_loss(new_mat, self.Ind_i, self.Ind_j, self.Ind_k).detach().item()
            upset_full = [upset.detach().item(), cycle_inconsistency_val]

            logstr += 'upset:,{:.6f}, cycle inconsistency:,{:.6f},'.format(upset.detach().item(), cycle_inconsistency_val)

            if self.args.SavePred:
                np.save(self.log_path + '/'+model_name+
                        '_scores'+str(split), score)
            
            print('Final results for {}:'.format(model_name))
            if self.label is not None:
                label_np = self.label_np
                # test
                mse1 = compErrorViaMatrices(score[test_index], label_np[test_index])
                outstrtest = 'Test MSE:, {:.3f}, '.format(mse1)
                MSE_full[split, 0] = [mse1]
                
                # val
                mse1 = compErrorViaMatrices(score[val_index], label_np[val_index])
                outstrval = 'Validation MSE:, {:.3f}, '.format(mse1)
                MSE_full[split, 1] = [mse1]
                
                
                # all
                mse1 = compErrorViaMatrices(score, label_np)
                outstrall = 'All MSE:, {:.3f},'.format(mse1)
                MSE_full[split, 2] = [mse1]
                    
                logstr += outstrtest + outstrval + outstrall

            print(logstr)

            with open(self.log_path + '/' + model_name + '_log'+str(split)+'.csv', 'a') as file:
                file.write(logstr)
                file.write('\n')
            # plot
            angles_diff = (score-label_np) % (2*np.pi)
            score = (score - angles_diff.mean()) % (2*np.pi)
            np.save('../uscities_pred/'+model_name+'_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed), score)
            for i in range(self.A.shape[0]):
                noisy_patch_i = whole_map[patch_indices[i]].copy()
                noisy_patch_i[:, 0] += added_noise_x[i]
                noisy_patch_i[:, 1] += added_noise_y[i]
                rotated_coordinates = np.dot(get_rotation_matrix(label_np[i]), noisy_patch_i.T).T
                rotated_coordinates = np.dot(get_rotation_matrix(-score[i]), rotated_coordinates.T).T
                plt.scatter(noisy_patch_i[:, 0], noisy_patch_i[:, 1], s=0.5, alpha=0.8, c='yellow')
                plt.scatter(rotated_coordinates[:, 0], rotated_coordinates[:, 1], s=0.5, alpha=0.8, c='blue')
            plt.scatter(whole_map[:, 0], whole_map[:, 1], s=1, c='red')
            plt.title('MSE={:.3f}'.format(MSE_full[split, 2, 0]))
            # plt.savefig('../uscities_plots/'+model_name+'_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed)+'.pdf',format='pdf')
            plt.savefig('../uscities_plots/'+model_name+'_k50_thres6_100eta'+str(int(100*args.eta))+args.outlier_style+'seed'+str(random_seed)+'.png',format='png')
            plt.show()
        return MSE_full, upset_full


# train and grap results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/'+args.dataset)


MSE_res = np.zeros([len(compare_names_all), args.num_trials*len(args.seeds), 3, 1])
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
    np.random.seed(random_seed)
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

np.save(os.path.join(dir_name,'MSE',method_str,save_name), MSE_res)
np.save(os.path.join(dir_name,'MSE_latest',method_str,save_name), MSE_res_latest)
np.save(os.path.join(dir_name,'upset',method_str,save_name), final_upset)
np.save(os.path.join(dir_name,'upset_latest',method_str,save_name), final_upset_latest)
