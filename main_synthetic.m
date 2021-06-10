clc;
clear;
close all;
addpath(genpath('algorithms'))
addpath(genpath('functions'))

% rng(10)
%% Problem Parameters
M = 25; % No of annotaors
K = 3;  % Rank
I = K*ones(1,M);  % the dimesnion of the tensor

N = 5000; % No of data items
p = 0.3;  % probbaility with which each annotator labels
p_super =0.8; % probbaility for super annotator
p_block=0.5; %probbaility with which each block A_iDA_j^T is observed

%% Impute parameters
% Method 0 : No imputation
%
% Method 1 : one super annottaor : an annottaor who has co-labeled with all
%            other annottaors sufficiently
% Method 2 : two super annottaors
%
% Method 3 : No superannottaors are known : so impute sufficient stastics.

impute_method =2;
% m_star_1 = 1; %index of super annottaor 1
% m_star_2 = 2; %index of super annottaor 2
percent_blocks_to_impute = 1; %How much imputation has to do
sample_thresh_imputation =10;
no_blocks_imputation=1000; %used for both method 2 & 3 imputation

%% Simulation Parameters
noise_flag=1;
n_iter = 5;
for s=1:n_iter
    %% Generate confusion matrices and prior
    A = generate_confusion_mat(M,K,-6); 
    lambda = ones(K,1);lambda = lambda./sum(lambda);

    %% Generate second order marginals
    if(noise_flag==0)
        opts={};
        m_star = randsample(M,2);m_star_1=m_star(1);m_star_2=m_star(2);
        opts.m_star_1=m_star_1;
        opts.m_star_2=m_star_2;
        opts.impute_method=impute_method;
        [M_mat,M_mat_true,N_valid_mat]=get_true_second_order_marginals(A,lambda,p_block,opts);
    else
        opts={};
        m_star = randsample(M,2);m_star_1=m_star(1);m_star_2=m_star(2);
        opts.m_star_1=m_star_1;
        opts.m_star_2=m_star_2;
        y = randsample(K,N,true,lambda); %generate ground-truth labels.
        p_obs = p*ones(M,1); 
        p_obs(opts.m_star_1)=p_super;
        p_obs(opts.m_star_2)=p_super;
        [F,f] = generate_annotator_labels(A,y,p_obs); %generate annotator estimates.
%         F = simulate_missing_blocks_mod2(f,K,p_block);
        [M_mat,N_valid_mat,M_tens,means_vec] = calc_annotator_moments(F,[2,3]); % generate second order statistics

        opts.impute_method=impute_method;
        opts.N_valid_mat=N_valid_mat;
        [M_mat,M_mat_unmask,M_mat_true,N_valid_mat] = mask_observed_marginals(M_mat,A,lambda,p_block,opts);
    end


    %% Run proposed algorithm coupled factorization
    flag_eigen_decomp = 0; % 1: eigen decomposition 0 : coupled factorization
    cf_init_via_eigendecomp=0; % if 1, coupled fact is initialized with eigen decomp
    ops={};
    ops.A_true=A;
    ops.l_true=lambda;
    ops.eigen_decomposition=flag_eigen_decomp;
    % imputation parameters
    ops.m_star_1=m_star_1;
    ops.m_star_2=m_star_2;
    ops.impute_method=impute_method;
    ops.percent_blocks_to_impute=percent_blocks_to_impute;
    ops.N_mat_valid = N_valid_mat > sample_thresh_imputation;
    ops.M_mat_true=M_mat_true;
    ops.no_blocks_imputation=no_blocks_imputation;
    ops.cf_init_via_eigendecomp=cf_init_via_eigendecomp;
    [A_est,l_est,time_total]=alg_SIngleEigDetMin(M_mat,K,ops);
    time_proposed(s) = time_total
    MSE_proposed(s) = getMSE(A_est,A,l_est,lambda)
    
     %% Run proposed algorithm - eigen decomposition
    flag_eigen_decomp = 1; % 1: eigen decomposition 0 : coupled factorization
    cf_init_via_eigendecomp=0; % if 1, coupled fact is initialized with eigen decomp
    ops={};
    ops.A_true=A;
    ops.l_true=lambda;
    ops.eigen_decomposition=flag_eigen_decomp;
    % imputation parameters
    ops.m_star_1=m_star_1;
    ops.m_star_2=m_star_2;
    ops.impute_method=impute_method;
    ops.percent_blocks_to_impute=percent_blocks_to_impute;
    ops.N_mat_valid = N_valid_mat > sample_thresh_imputation;
    ops.M_mat_true=M_mat_true;
    ops.no_blocks_imputation=no_blocks_imputation;
    ops.cf_init_via_eigendecomp=cf_init_via_eigendecomp;
    [A_est1,l_est1,time_total]=alg_SIngleEigDetMin(M_mat,K,ops);
    time_proposed1(s) = time_total
    MSE_proposed1(s) = getMSE(A_est1,A,l_est1,lambda)
    
    %% Run proposed coupled factorization+D&S algorithm
    opts.Nround=10;
    a=tic;
    [A_est_ds,l_est_ds]=EM_yeredor(A_est,l_est,f,opts);
    time_proposed_ds(s) = toc(a);
    MSE_proposed_ds(s) = getMSE(A_est_ds,A,l_est_ds,lambda)
    
    %% Run proposed eigen decomposition+D&S algorithm
    opts.Nround=10;
    a=tic;
    [A_est_ds1,l_est_ds1]=EM_yeredor(A_est1,l_est1,f,opts);
    time_proposed_ds1(s) = toc(a);
    MSE_proposed_ds1(s) = getMSE(A_est_ds1,A,l_est_ds1,lambda)
    
    %%  Run MultiSPA
    a=tic;
    [A_est_mspa,l_est_mpsa] = MultiSPA(M_mat,K,0);
    time_mspa(s) = toc(a);
    MSE_mspa(s) = getMSE(A_est_mspa,A,l_est_mpsa,lambda)
    
    %% Run MultiSPA-D&S
    opts.Nround=10;
    a=tic;
    [A_est_mspa_ds,l_est_mspa_ds] = EM_yeredor(A_est_mspa,l_est_mpsa,f,opts);
    time_mspa_ds(s) = toc(a);
    MSE_mspa_ds(s) = getMSE(A_est_mspa_ds,A,l_est_mspa_ds,lambda)
    
    %% Run MultiSPA-KL
    marg = combnk(1:M,2);           % marg defines the pairs of variables (or triples 2->3)
    marg = num2cell(marg,2);        % convert it to cell
    Y=get_second_order_stat(M_mat,marg);
    opts = {}; 
    opts.marg = marg; 
    opts.max_iter = 500; 
    opts.tol_impr = 1e-6;
    opts.A0 = A_est_mspa; 
    opts.l0 = l_est_mpsa;
    I=K*ones(1,M);
    [A_est_mspa_kl,l_est_mspa_kl,Out] = N_CTF_AO_KL(Y,I,K,opts);
    MSE_mspa_kl(s) = getMSE(A_est_mspa_kl,A,l_est_mspa_kl,lambda)
    time_mspa_kl(s) = sum(Out.time_stamps);
%     
    %% Run TensorADMM
    params.inner_max_iter = 10; %inner ADMM max iterations
    params.outer_max_iter = 500; %outer loop max iterations
    params.outer_tol = 1e-6; %outer loop tolerance
    params.inner_tol = 1e-3; %inner loop tolerance
    params.init = 0; %Algorithm initialization. Set to 1 for better than random.
    params.display = 0;

    a = tic;
    [A_est_tensor,pi_vec_tensor] = EstConfMat_AO_ADMM(means_vec,M_mat,M_tens,params);
    time_tensor(s)= toc(a);
    MSE_tensor(s) = getMSE(A_est_tensor,A,pi_vec_tensor,lambda)
 
    
    %% Run Spectral-D&S
    opts.Nround=10;
    a=tic;
    [rate_err_emspec(s),A_est_spec_ds] = run_EM_Spectral(f,K,y,opts.Nround);
    l_est_spec_ds  = ones(K,1)/K;
    time_spec_ds(s) = toc(a);
    MSE_spec_ds(s) = getMSE(A_est_spec_ds,A,l_est_spec_ds,lambda)
    
    %% Run MV-D&S
    opts.Nround=10;
    a=tic;
    [~,A_est_mv_ds] = run_EM_MV(f,K,y,opts.Nround);
    l_est_mv_ds  = ones(K,1)/K;
    time_mv_ds=toc(a);
    MSE_mv_ds(s) = getMSE(A_est_mv_ds,A,l_est_mv_ds,lambda)
%     
 
end
mean_MSE_proposed=mean(MSE_proposed)
mean_MSE_proposed1=mean(MSE_proposed1)
mean_MSE_proposed_ds=mean(MSE_proposed_ds)
mean_MSE_proposed_ds1=mean(MSE_proposed_ds1)
mean_MSE_mspa=mean(MSE_mspa)
mean_MSE_mspa_ds=mean(MSE_mspa_ds)
mean_MSE_mspa_kl=mean(MSE_mspa_kl)
mean_MSE_tensor=mean(MSE_tensor)
mean_MSE_spec_ds=mean(MSE_spec_ds)
mean_MSE_mv_ds=mean(MSE_mv_ds)

mean_time_proposed=mean(time_proposed)
mean_time_proposed1=mean(time_proposed1)
mean_time_proposed_ds=mean(time_proposed_ds)
mean_time_proposed_ds1=mean(time_proposed_ds1)
mean_time_mspa=mean(time_mspa)
mean_time_mspa_ds=mean(time_mspa_ds)
mean_time_mspa_kl=mean(time_mspa_kl)
mean_time_tensor=mean(time_tensor)
mean_time_spec_ds=mean(time_spec_ds)
mean_time_mv_ds=mean(time_mv_ds)

rows = {'MSE';'STD_MSE';'time(s)'};
 Proposed_CF = [nanmean(MSE_proposed);std(MSE_proposed);mean(time_proposed,"all")];
 Proposed_CF_EM = [nanmean(MSE_proposed_ds);std(MSE_proposed_ds);mean(time_proposed_ds,"all")];
 Proposed_SVDEig = [nanmean(MSE_proposed1);std(MSE_proposed1);mean(time_proposed1,"all")];
 Proposed_SVDEig_EM = [nanmean(MSE_proposed_ds1);std(MSE_proposed_ds1);mean(time_proposed_ds1,"all")];
 MultiSPA = [nanmean(MSE_mspa);std(MSE_mspa);mean(time_mspa,"all")];
 MultiSPA_EM = [nanmean(MSE_mspa_ds);std(MSE_mspa_ds);mean(time_mspa_ds,"all")];
 MultiSPA_KL  = [nanmean(MSE_mspa_kl);std(MSE_mspa_kl);mean(time_mspa_kl,"all")];
 TensorADMM = [nanmean(MSE_tensor);std(MSE_tensor);mean(time_tensor,"all")];
 Spectral_DS= [nanmean(MSE_spec_ds);std(MSE_spec_ds);mean(time_spec_ds,"all")];
 MV_DS= [nanmean(MSE_mv_ds);std(MSE_mv_ds);mean(time_mv_ds,"all")];


 T = table(Proposed_CF,Proposed_CF_EM,Proposed_SVDEig,Proposed_SVDEig_EM,MultiSPA,...
     MultiSPA_EM,MultiSPA_KL,TensorADMM,Spectral_DS,MV_DS,...
    'RowNames',rows);

writetable(T,'table_synthetic_M_25_S_5000_des.csv','WriteRowNames',true)
