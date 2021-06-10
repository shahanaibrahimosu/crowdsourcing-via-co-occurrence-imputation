clc;
clear;
close all;
addpath(genpath('algorithms'))
addpath(genpath('functions'))
addpath(genpath('dataset'))


%% Simulation Parameters
n_iter = 10;
M = 10; %number of annotators.
K = 3; %number of classes. 
fix_permutation=1;

%% Probability of blocks observed
p_block=0.5;

%% Imputation parameters
impute_method =2;
%m_star_1 = 2; %index of super annottaor 1
%m_star_2 = 4; %index of super annottaor 2
percent_blocks_to_impute = 1; %How much imputation has to do
sample_thresh_imputation =5;
no_blocks_imputation=10; %used for both method 2 & 3 imputation



%%  Run the simulation
for s=1:n_iter
%   the probability of labelling for each annotator 
      p_vec = 0.5+(0.7-0.5)*rand(M,1);
    m_star = randsample(M,2);m_star_1=m_star(1);m_star_2=m_star(2);
%      p_vec = (0.2)*rand(M,1);
     p_vec(m_star_1)=0.8; % if super annotators are present, higher prob. can be given
     p_vec(m_star_2)=0.8; 
      p_vec = 1*ones(M,1);
    %% Generate the data
    [Data,I]    = LoadDataset_connect4();
    [rows, col] = size(Data);
    [trainInd,testInd,~] = dividerand(rows,0.5,0.5,0);
    Data = Data(trainInd,:);
    [f,F,Data] = get_real_annotations_connect4(Data,K,M,p_vec,p_block);

    N = size(Data,1);
    y = Data(:,1);
    pi_vec=ones(K,1)/K;
    
    min_res=0;
    max_res=0;
    error_resp=zeros(1,M);
    if(1)
        for m=1:M
            val_index = f(m,:)~=0;
            resp=f(m,val_index)';
            error_resp(m)=sum(y(val_index)~=resp)/length(resp)
        end
        min_res=min_res+min(error_resp);
        max_res=max_res+max(error_resp);
    end
    
    
    %% Get second order statistics from annotator data
    [M_mat,N_valid_mat,M_tens,means_vec] = calc_annotator_moments(F,[2,3]);
    opts={};    
    opts.m_star_1=m_star_1;
    opts.m_star_2=m_star_2;
    opts.impute_method=impute_method;
    opts.N_valid_mat=N_valid_mat;
    A=[];lambda=[];
    [M_mat,M_mat_unmask,M_mat_true,N_valid_mat] = mask_observed_marginals(M_mat,A,lambda,1,opts);
   
    
    %% Run Proposed (CF)
    ops={};
    ops.eigen_decomposition=0;
    % imputation parameters
    ops.m_star_1=m_star_1;
    ops.m_star_2=m_star_2;
    ops.impute_method=impute_method;
    ops.percent_blocks_to_impute=percent_blocks_to_impute;
    ops.N_mat_valid = N_valid_mat > sample_thresh_imputation;
    ops.M_mat_true=[];
    ops.no_blocks_imputation=no_blocks_imputation;
    ops.cf_init_via_eigendecomp=0;
    a=tic;
    [A_est,l_est]=alg_SIngleEigDetMin_mod(M_mat,K,ops);
    [~,A_est] = getPermutedMatrix(A_est,1:M);
    time_proposed1(s) = toc(a);

    % Estimate Labels
    l_est=ones(K,1)/K;
    idx = label_estimator(A_est,F,'MAP',l_est); 

    % Estimate label accuracy
    [err_proposed1(s),rate_err_proposed1(s)] = label_accuracy(idx,y);
    disp(['Proposed-CF(MAP) - Label Errors: ',num2str(err_proposed1(s))]);
    disp(['Proposed-CF(MAP) - Error Rate:  ',num2str(rate_err_proposed1(s))]);

    %% Run Proposed-EM (CF)
    opts.Nround=10;
    a=tic;
    [A_est_ds,l_est_ds]=EM_yeredor(A_est,l_est,f,opts);
    time_proposed_ds1(s) = toc(a);

    % Estimate Labels
    idx = label_estimator(A_est_ds,F,'MAP',l_est_ds); 

    % Estimate label accuracy
    [err_proposed_ds1(s),rate_err_proposed_ds1(s)] = label_accuracy(idx,y);
    disp(['Proposed-CF-EM(MAP) - Label Errors: ',num2str(err_proposed_ds1(s))]);
    disp(['Proposed-CF-EM(MAP) - Error Rate:  ',num2str(rate_err_proposed_ds1(s))]);
    
    %% Run Proposed (Eigen Decomposition)
    ops={};
    ops.eigen_decomposition=1;
    % imputation parameters
    ops.m_star_1=m_star_1;
    ops.m_star_2=m_star_2;
    ops.impute_method=impute_method;
    ops.percent_blocks_to_impute=percent_blocks_to_impute;
    ops.N_mat_valid = N_valid_mat > sample_thresh_imputation;
    ops.M_mat_true=[];
    ops.no_blocks_imputation=no_blocks_imputation;
    ops.cf_init_via_eigendecomp=0;
    a=tic;
    [A_est,l_est]=alg_SIngleEigDetMin(M_mat,K,ops);
    [~,A_est] = getPermutedMatrix(A_est,1:M);
    time_proposed2(s) = toc(a);

    % Estimate Labels
    l_est=ones(K,1)/K;
    idx = label_estimator(A_est,F,'MAP',l_est); 

    % Estimate label accuracy
    [err_proposed2(s),rate_err_proposed2(s)] = label_accuracy(idx,y);
    disp(['Proposed-SVDEigen(MAP) - Label Errors: ',num2str(err_proposed2(s))]);
    disp(['Proposed-SVDEigen(MAP) - Error Rate:  ',num2str(rate_err_proposed2(s))]);

    %% Run Proposed-EM (eigen_decomp)
    opts.Nround=10;
    a=tic;
    [A_est_ds,l_est_ds]=EM_yeredor(A_est,l_est,f,opts);
    time_proposed_ds2(s) = toc(a);

    % Estimate Labels
    idx = label_estimator(A_est_ds,F,'MAP',l_est_ds); 

    % Estimate label accuracy
    [err_proposed_ds2(s),rate_err_proposed_ds2(s)] = label_accuracy(idx,y);
    disp(['Proposed-SVDEigen-EM(MAP) - Label Errors: ',num2str(err_proposed_ds2(s))]);
    disp(['Proposed-SVDEigen-EM(MAP) - Error Rate:  ',num2str(rate_err_proposed_ds2(s))]);


   %% Run MultiSPA
    a=tic;
    [A_est_mspa,pi_vec_est] = MultiSPA(M_mat,K,fix_permutation);
    time_mspa(s)=toc(a);

    % Estimate Labels
    idx = label_estimator(A_est_mspa,F,'MAP',pi_vec_est); 

    % Estimate label accuracy
    [n_err_mspa,rate_err_mspa(s)] = label_accuracy(idx,y);
    disp(['MultiSPA(MAP) - Label Errors: ',num2str(n_err_mspa)]);
    disp(['MultiSPA(MAP) - Error rate:  ',num2str(rate_err_mspa(s))]);
    
    %% Run MultiSPA-KL
    marg = combnk(1:M,2);           % marg defines the pairs of variables (or triples 2->3)
    marg = num2cell(marg,2);        % convert it to cell
    Y=get_second_order_stat(M_mat,marg);
    opts = {}; 
    opts.marg = marg; 
    opts.max_iter = 100; 
    opts.tol_impr = 1e-6;
    opts.A0 = A_est_mspa; 
    opts.l0 = pi_vec_est;
    I=K*ones(1,M);
    [A_est_mspa_kl,l_est_mspa_kl,Out] = N_CTF_AO_KL(Y,I,K,opts);
    time_mspa_kl(s) = sum(Out.time_stamps);
    
    % Estimate Labels
    idx = label_estimator(A_est_mspa_kl,F,'MAP',l_est_mspa_kl); 

    % Estimate label accuracy
    [n_err_mspa_kl,rate_err_mspa_kl(s)] = label_accuracy(idx,y);
    disp(['MultiSPA-KL(MAP) - Label Errors: ',num2str(n_err_mspa_kl)]);
    disp(['MultiSPA-KL(MAP) - Error rate:  ',num2str(rate_err_mspa_kl(s))]);


    %% Run MultiSPA-EM
    opts.Nround=10;
    a=tic;
    [A_est,pi_vec_est] = EM_yeredor(A_est_mspa,pi_vec,f,opts);
    time_mspaem(s)=toc(a);

    % Estimate Labels
    idx = label_estimator(A_est,F,'MAP',pi_vec_est); 

    % Estimate label accuracy
    [n_err_em,rate_err_mspaem(s)] = label_accuracy(idx,y);
    disp(['MultiSPA-EM(MAP) - Label Errors: ',num2str(n_err_em)]);
    disp(['MultiSPA-EM(MAP) - Error rate:  ',num2str(rate_err_mspaem(s))]);

    %% Run TensorADMM
    params.inner_max_iter = 10; %inner ADMM max iterations
    params.outer_max_iter = 500; %outer loop max iterations
    params.outer_tol = 1e-6; %outer loop tolerance
    params.inner_tol = 1e-3; %inner loop tolerance
    params.init = 0; %Algorithm initialization. Set to 1 for better than random.
    params.display = 0;
    
    for i=1:length(M_mat)
        for j=1:length(M_mat)
            if(isempty(M_mat{i,j}))
                M_mat{i,j}=zeros(K,K);
            end
        end
    end

    a = tic;
    [A_est_tensor,pi_vec_tensor] = EstConfMat_AO_ADMM(means_vec,M_mat,M_tens,params);
    [~,A_est_tensor] = getPermutedMatrix(A_est_tensor,1:M);
    time_tensor(s)= toc(a);

    % Estimate Labels
    idx = label_estimator(A_est_tensor,F,'MAP',pi_vec_tensor); 

    % Estimate label accuracy
    [n_err_tensor,rate_err_tensor(s)] = label_accuracy(idx,y);
    disp(['TensorADMM - Label Errors: ',num2str(n_err_tensor)]);
    disp(['TensorADMM - Error rate:  ',num2str(rate_err_tensor(s))]);



    %% Run EM algorithm - Spectral
    opts.Nround=10;
    a=tic;
    [rate_err_emspec(s),A_est] = run_EM_Spectral(f,K,y,opts.Nround);
    time_spectral_em(s)=toc(a);
    err_em = round(rate_err_emspec(s)*length(find(y>0)));
    % Estimate label accuracy
    disp(['Spectral D&S - Label Errors: ',num2str(err_em)]);
    disp(['Spectral D&S - Error rate:  ',num2str(rate_err_emspec(s))]);

    %% Run EM algorithm - MV
    opts.Nround=10;
    a=tic;
    [rate_err_emmv(s),A_est] = run_EM_MV(f,K,y,opts.Nround);
    [~,A_est] = getPermutedMatrix(A_est,1:M);
    time_mvds(s)=toc(a);

    err_em = round(rate_err_emmv(s)*length(find(y>0)));
    % Estimate label accuracy
    disp(['MV EM - Label Errors: ',num2str(err_em)]);
    disp(['MV EM - Error rate:  ',num2str(rate_err_emmv(s))]);
    
 
%     %% Minimax Entropy  
%     L = f';
%     true_labels=y';
%     Model = crowd_model(L, 'true_labels',true_labels);
%     disp('***************Running Minimax Entropy ****************************');
%     a = tic;
%     lambda_worker = 0.25*Model.Ndom^2; lambda_task = lambda_worker * (mean(Model.DegWork)/mean(Model.DegTask)); % regularization parameters
%     opts={'lambda_worker', lambda_worker, 'lambda_task', lambda_task, 'maxIter',50,'TOL',5*1e-3','verbose',1};
%     result1 =  MinimaxEntropy_crowd_model(Model,'algorithm','categorical',opts{:}); 
%     rate_err_minmax(s) = result1.error_rate;
%     n_err_minmax = ceil(result1.error_rate*N);
%     time_minmax = toc(a);
%     
%      % Estimate label accuracy
%     disp(['Minimax Entropy - Label Errors: ',num2str(n_err_minmax)]);
%     disp(['Minimax Entropy - Error rate:  ',num2str(rate_err_minmax(s))]);


    %% KOS
    a = tic;
    [A1] = convert_for_comp(f);
    valid_index = find(y>0);
    [error_KOS] = KOS(A1,y,valid_index,N,K,M);
    n_err_KOS = ceil(error_KOS(end)*N);
    rate_err_KOS(s) = error_KOS(end);
    time_KOS = toc(a);
    
     % Estimate label accuracy
    disp(['KOS - Label Errors: ',num2str(n_err_KOS)]);
    disp(['KOS - Error rate:  ',num2str(rate_err_KOS(s))]);


    %% GhoshSVD
    a = tic;
    [A1] = convert_for_comp(f);
    valid_index = find(y>0);
    [error_GhoshSVD] = GhostSVD(A1,y,valid_index,N,K,M);
    n_err_GhoshSVD = ceil(error_GhoshSVD(end)*N);
    rate_err_ghoshsvd(s) = error_GhoshSVD(end);
    time_ghoshsvd(s) = toc(a);

    % Estimate label accuracy
    disp(['GhoshSVD - Label Errors: ',num2str(n_err_GhoshSVD)]);
    disp(['GhoshSVD - Error rate:  ',num2str(rate_err_ghoshsvd(s))]);

    %% EIgenRatio
    a = tic;
    [A1] = convert_for_comp(f);
    valid_index = find(y>0);
    [error_RatioEigen] = EigenRatio(A1,y,valid_index,N,K,M);
    n_err_eigenratio = ceil(error_RatioEigen(end)*N);
    rate_err_eigenratio(s) = error_RatioEigen(end);
    time_eigenratio = toc(a);
    
    % Estimate label accuracy
    disp(['EIgenRatio - Label Errors: ',num2str(n_err_eigenratio)]);
    disp(['EIgenRatio - Error rate:  ',num2str(rate_err_eigenratio(s))]);

    %% Majority Voting
    f2 = f;
    f2(f2 == 0) = NaN; %convert 0's to NaN as they are ignored by the mode function
    idx_MV = mode(f2); %estimate labels using majority voting
    u = find(idx_MV(valid_index)~=y(valid_index)');
    F_tensor = zeros(K,N,M);
    for i=1:M
        F_tensor(:,:,i) = F{i};
    end
    [rate_err_mv(s)] = majority_voting_label(F_tensor,y,valid_index);
    n_err_mv = round(rate_err_mv(s)*length(valid_index));
    time_mv(s)=0;
    % Estimate label accuracy
    disp(['Majority Voting - Label Errors: ',num2str(n_err_mv)]);
    disp(['Majority Voting - Error rate:  ',num2str(rate_err_mv(s))]);
end

rows = {'Error Rate (p=0.5_7)';'std_dev';'time(s)'};
 Proposed_CF = [nanmean(rate_err_proposed1);std(rate_err_proposed1);mean(time_proposed1,"all")];
 Proposed_CF_EM = [nanmean(rate_err_proposed_ds1);std(rate_err_proposed_ds1);mean(time_proposed_ds1,"all")];
 Proposed_SVDEig = [nanmean(rate_err_proposed2);std(rate_err_proposed2);mean(time_proposed2,"all")];
 Proposed_SVDEig_EM = [nanmean(rate_err_proposed_ds2);std(rate_err_proposed_ds2);mean(time_proposed_ds2,"all")];
 MultiSPA = [nanmean(rate_err_mspa);std(rate_err_mspa);mean(time_mspa,"all")];
 MultiSPA_EM = [nanmean(rate_err_mspaem);std(rate_err_mspaem);mean(time_mspaem,"all")];
 MultiSPA_KL  = [nanmean(rate_err_mspa_kl);std(rate_err_mspa_kl);mean(time_mspa_kl,"all")];
 TensorADMM = [nanmean(rate_err_tensor);std(rate_err_tensor);mean(time_tensor,"all")];
 Spectral_DS= [nanmean(rate_err_emspec);std(rate_err_emspec);mean(time_spectral_em,"all")];
 MV_DS= [nanmean(rate_err_emmv);std(rate_err_emmv);mean(time_mvds,"all")];
% Minmax_Entropy=[nanmean(rate_err_minmax);std(rate_err_minmax);mean(time_minmax,"all")];
KOS=[nanmean(rate_err_KOS);std(rate_err_KOS);mean(time_KOS,"all")];
GhoshSVD=[nanmean(rate_err_ghoshsvd);std(rate_err_ghoshsvd);mean(time_ghoshsvd,"all")];
EigenRatio= [nanmean(rate_err_eigenratio);std(rate_err_eigenratio);mean(time_eigenratio,"all")];
Majority_Voting= [nanmean(rate_err_mv);std(rate_err_mv);mean(time_mv,"all")];

T = table(Proposed_CF,Proposed_CF_EM,Proposed_SVDEig,Proposed_SVDEig_EM,MultiSPA,...
     MultiSPA_EM,MultiSPA_KL,TensorADMM,Spectral_DS,MV_DS,...
    KOS,GhoshSVD,EigenRatio,Majority_Voting,'RowNames',rows);

writetable(T,'table_connect4_5_7.csv','WriteRowNames',true)



