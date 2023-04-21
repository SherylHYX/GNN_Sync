cd ../src

../../parallel -j10 --resume-failed --results ../Output/ksync_GNNSync --joblog ../joblog/ksync_GNNSync CUDA_VISIBLE_DEVICES=0 python ./ksync_train.py --optimizer SGD   --hidden 8 --spectral_step_num 5 --outlier_style {1} --eta {2} --p {3} --N 360 --upset_coeff {4} --cycle_coeff {5} --sync_baseline row_norm_spectral --dataset {6} --k {7} --lr 0.005 --all_methods GNNSync --reg_coeff 1 ::: gamma multi_normal0 multi_normal1 block_normal6 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 ::: 0.05 0.1 0.15 ::: 1 0 ::: 1 0 ::: BAO RGGO ERO ::: 4 3 2