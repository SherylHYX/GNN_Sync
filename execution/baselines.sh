cd ../src

../../parallel -j15 --resume-failed --results ../Output/sync_baselines --joblog ../joblog/sync_baselines python ./train.py --p {1} --eta {2} --no-cuda --outlier_style {3} --N 360 --dataset {4} --all_methods {5} ::: 0.05 0.1 0.15 ::: 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 ::: gamma ::: RGGO ERO BAO ::: spectral row_norm_spectral GPM TranSync CEMP_GCW CEMP_MST trivial

../../parallel -j15 --resume-failed --results ../Output/ksync_baselines --joblog ../joblog/ksync_baselines python ./ksync_train.py  --p {1} --eta {2} --no-cuda --outlier_style {3} --N 360 --dataset {4} --all_methods {5} --k {6} ::: 0.05 0.1 0.15 ::: 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0 ::: gamma gamma_diff ::: RGGO ERO BAO ::: spectral row_norm_spectral trivial ::: 2 3 4