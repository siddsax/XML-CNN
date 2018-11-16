addpath('/scratch/work/saxenas2/fastxml/manik/Tools/matlab/')
addpath('/scratch/work/saxenas2/fastxml/manik/tools/')
addpath('/scratch/work/saxenas2/fastxml/manik/Tools/metrics/')
addpath('/scratch/work/saxenas2/fastxml/manik/FastXML/')

A = .55;
B = 1.5;

load score_matrix.mat
[I, J, S] = find(score_matrix);
[sorted_I, idx] = sort(I);
J = J(idx);
S = S(idx);
score_matrix = sparse(J, sorted_I, S);

load ty.mat
[I, J, S] = find(ty);
[sorted_I, idx] = sort(I);
J = J(idx);
S = S(idx);
ty = sparse(J, sorted_I, S);
ip = inv_propensity(ty,A,B);

[metrics] = get_all_metrics(score_matrix , ty, ip)
disp(metrics)

% -------- For RCV1 His neural net--------

% prec 96.58 89.82 79.66 65.28 55.15
% nDCG 96.58 92.51 90.96 91.01 91.46
% prec_wt 86.22 86.25 87.38 87.70 88.48
% nDCG_wt 86.22 86.24 87.00 87.21 87.65

% -----------------------------------------

% prec 93.26 86.08 75.64 62.28 52.79
% nDCG 93.26 88.84 86.81 87.18 87.84
% prec_wt 73.04 76.45 78.40 80.02 81.59
% nDCG_wt 73.04 75.62 77.04 78.06 78.96

% prec 95.50 87.29 76.72 63.20 53.59
% nDCG 95.50 90.29 88.17 88.53 89.18
% prec_wt 72.24 76.67 79.44 81.27 82.96
% nDCG_wt 72.24 75.59 77.59 78.76 79.73


% ---------- Initialized weights with Dropouts -------------
%  Best for test -------------------
% prec 94.06 84.04 73.35 60.90 51.89
% nDCG 94.06 87.45 84.92 85.63 86.51
% prec_wt 70.89 73.01 74.81 77.17 79.28
% nDCG_wt 70.89 72.50 73.76 75.21 76.40

%  Best for train -------------------
% prec 93.62 84.88 74.66 61.41 52.02
% nDCG 93.62 88.00 86.00 86.34 86.98
% prec_wt 71.90 75.07 77.10 78.54 80.01
% nDCG_wt 71.90 74.30 75.76 76.67 77.52


% ---------------- base_model_with_test_saving_after_each_run ------
% model_best_batch
% prec 94.49 86.20 75.71 62.40 52.84
% nDCG 94.49 89.23 87.11 87.53 88.16
% prec_wt 72.40 76.11 78.32 80.02 81.60
% nDCG_wt 72.40 75.21 76.81 77.88 78.79

% model_best_for_test
% prec 94.98 86.05 75.65 62.45 53.06
% nDCG 94.98 89.21 87.08 87.54 88.29
% prec_wt 71.91 75.42 77.85 79.69 81.58
% nDCG_wt 71.91 74.57 76.30 77.47 78.54


%  --------------- L1 loss ----------------------
model_best_for_test
bad!!!

model_best_batch
bad!!!

% ------------------ Ablation --------------
% prec 94.59 87.66 77.32 63.61 53.84
% nDCG 94.59 90.38 88.51 88.81 89.37
% prec_wt 74.26 77.99 80.16 81.68 83.20
% nDCG_wt 74.26 77.08 78.66 79.63 80.50