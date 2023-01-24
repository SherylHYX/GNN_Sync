import argparse
import os

import torch

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Group synchronization.")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--debug', '-D',action='store_true', default=False,
                        help='Debugging mode, minimal setting.')
    parser.add_argument('--seed', type=int, default=31, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='(Initial) learning rate for spectral step.')
    parser.add_argument('--separate_graphs',action='store_true', default=False,
                        help='Whether not to use the same H for k>1 when applying projected gradient steps.')
    parser.add_argument('--trainable_alpha',action='store_true', default=False,
                        help='Whether to set the spectral step learning rate to be trainable.')
    parser.add_argument('--spectral_step_num', type=int, default=5,
                        help='The number of spectral steps.')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer to use. Adam or SGD in our case.')
    parser.add_argument('--sync_baseline', type=str, default='row_norm_spectral',
                        help='The baseline model used for obtaining baseline result.')
    parser.add_argument("--all_methods",
                        nargs="+",
                        type=str,
                        help="Methods to use.")
    parser.set_defaults(all_methods=['GNNSync'])
    parser.add_argument("--seeds",
                        nargs="+",
                        type=int,
                        help="seeds to generate random graphs.")
    parser.set_defaults(seeds=[10, 20, 30, 40, 50])

    # synthetic model hyperparameters below
    parser.add_argument('--p', type=float, default=0.05,
                        help='Edge probability parameter.')
    parser.add_argument('--N', type=int, default=360,
                        help='Number of nodes in the synthetic model.')
    parser.add_argument('--k', type=int, default=1,
                        help='Value k for k-synchronization.')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.') 
    parser.add_argument('--num_trials', type=int, default=2,
                        help='Number of trials to generate results.')      
    parser.add_argument('--outlier_style', type=str, default='gamma',
                        help='Outlier model rating style, gamma_diff or gamma.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Noise level parameter.')
    parser.add_argument('--upset_coeff', type=float, default=1.0,
                        help='Coefficient of upset loss.')
    parser.add_argument('--cycle_coeff', type=float, default=0,
                        help='Coefficient of cycle inconsistency loss.')
    parser.add_argument('--reg_coeff', type=float, default=1,
                        help='Coefficient of the variance regularization term in cycle inconsistency loss.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Number of iterations to consider for early stopping.')
    parser.add_argument('--regenerate_data', action='store_true', help='Whether to force creation of data.')
    parser.add_argument('--load_only', action='store_true', help='Whether not to store generated data.')
    parser.add_argument('-SavePred', '-SP', action='store_true', help='Whether to save predicted labels.')
    parser.add_argument('--log_root', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../logs/'), 
                        help='The path saving model.t7 and the training process')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/'), 
                        help='Data set folder.')
    parser.add_argument('--dataset', type=str, default='ERO/', help='Data set selection.')
    
    args = parser.parse_args()
    if args.dataset[-1]!='/':
        args.dataset += '/'
    
    if args.dataset[:3] in ['ERO', 'BAO', 'RGG']:
        default_name_base = 'p' + str(int(100*args.p)) + 'N' + str(args.N)
        default_name_base += 'eta' + str(int(100*args.eta)) + 'style' + str(args.outlier_style)
        if args.dataset[:4] == 'RGGO':
            args.dataset = 'RGGO/' + default_name_base
        else:
            args.dataset = args.dataset[:3] + '/' + default_name_base

    if args.k > 1 and args.dataset[:3] in ['ERO', 'BAO', 'RGG']:
        args.dataset += 'k' + str(args.k)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    if args.debug:
        args.num_trials = 2
        args.seeds=[10]
        args.epochs = 4
        args.pretrain_epochs = 1
        args.log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../debug_logs/')
    return args
