function [Gamma_hat,p_vec_hat] = EstConfMat_AO_ADMM(M1_vec,M2_mat,M3_tens,params)
%ESTCONFMAT_AO_ADMM Estimate KxK (K=number of labels) Confusion matrices
%for M annotators, as well as the class prior probabilities, 
%using joint matrix-tensor factorization and ADMM.
%   Note: requires Tensor Toolbox: http://www.sandia.gov/~tgkolda/TensorToolbox
%
%   Input: M1_vec - Mx1 cell containing mean responses (Kx1) of annotators.
%          M2_mat - MxM cell containing KxK crosscorrelation matrices between annotator
%          responses
%              i.e. If we are given crosscorrelation matrices then  M_mat{i,j} = E[f_i(X)f_j(X)^T] where E stands for
%              expectation and f_i is the classification function of the
%              i-th annotator.
%         M3_tens - MxMxM cell containing KxKxK crosscorrelation tensors
%         between annotator responses
%          params - struct containing hyperparameters for the algorithm
%                   params.inner_max_iter - number of maximum inner loop ADMM
%                   iterations
%                   params.inner_tol - tolerance of inner ADMM problem
%                   params.outer_tol - tolerance of the entire algorithm
%                   params.outer_max_iter - maximum number of outer
%                   iterations
%                   params.display - 0/1 flag. Set =1 if you wish to see
%                   error graphs
%                   params.init - 0/1 flag. Set =1 to initialize confusion
%                   matrices s.t. annotators are better than random
%
%
%    Output: Gamma_hat - Mx1 cell containing estimated KxK confusion matrices
%    for each annotator
%          p_vec_hat - Kx1 vector containing class prior probabilities
%     Panagiotis Traganitis. traga003@umn.edu
%
%%%%%%%%%%%%%%%%%%%%%POSSIBLY UNFINISHED%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Current version requires too much memory if M or K get too large!
%
%

if nargin < 3
    error('1st order moments and 2nd and 3rd order cross-moments are required for this function');
end
if nargin < 4
    warning('No hyperparameters provided - reverting to defaults');
    params.inner_mu = 10;
    max_iter = 1000;
    params.inner_max_iter = 10;
    params.inner_tol = 1e-3;
    %params.inner_lambda = 1e-1;
    params.inner_mu_decrease = 1e-3;
    tol = 1e-3;
    display = 0;
else
    max_iter = params.outer_max_iter;
    tol = params.outer_tol;
    display = params.display;
    M = size(M2_mat,1);
    if ~isfield(params,'inner_mu_decrease')
        params.inner_mu_decrease = 0;
    end
%     if ~isfield(params,'inner_lambda_pi')
%         params.inner_lambda_pi = params.inner_lambda;
%     end
    if ~isfield(params,'m_vec_val')
        params.m_vec_val = ones(M,1);
    end
    if ~isfield(params,'M_mat_val')
       params.M_mat_val = ones(M,M); 
    end
    if ~isfield(params,'M_tens_val')
        params.M_tens_val = ones(M,M,M);
    end
    if ~isfield(params,'p_obs')
        params.p_obs = ones(M,1);
    end
    if ~isfield(params,'inner_max_iter_pi')
        params.inner_max_iter_pi = params.inner_max_iter;
    end
    if ~isfield(params,'fixed_pi_vec')
        params.fixed_pi = 0;
    else
        params.fixed_pi = 1;
    end
    if ~isfield(params,'fixed_mu')
        fixedmu = false;
    else
        fixedmu = true;
        params.inner_mu = params.fixed_mu;
    end
    if ~isfield(params,'mu_const')
        mu_const1 = 1e-7;
        mu_const2 = 1e-2;
    else
        mu_const1 = params.mu_const.const1;
        mu_const2 = params.mu_const.const2;
    end
end

obj_fun_val = zeros(params.outer_max_iter,1);
M = size(M2_mat,1);
K = length(M1_vec(:,1));
if nargin < 4
    params.M_mat_val = ones(M,M);
    params.M_tens_val = ones(M,M,M);
    params.p_obs = ones(M,1);
end

params.K = K; params.M = M;

%%%%%%%%%%% preprocess data %%%%%%%%%%%%%%%%%%%%%%%%
disp('preprocessing data');
Data_struct.M3_mat = cell(M,M,M); %this process can probably be done more efficiently
Data_struct.M3_vec = cell(M,M,M); %currently it requires too much memory
%Data_struct.M2_vec = cell(M,M);
for i=1:M
    for j=1:M
        if j~=i
            %Data_struct.M2_vec{i,j} = reshape(M2_mat{i,j},[K*K 1]);
            for k=1:M
                if k~=i && k~=j
                    tmpmat = zeros(K*K,K);
                    for kk = 1:K
                        tmpmat(:,kk) = reshape(M3_tens{i,j,k}(kk,:,:),[K*K 1]);
                    end
                    Data_struct.M3_mat{i,j,k} = tmpmat;
                    %Data_struct.M3_mat{i,j,k} = reshape(M3_tens{i,j,k},[K*K K]);
                    %Data_struct.M3_vec{i,j,k} = reshape(Data_struct.M3_mat{i,j,k},[K*K*K 1]);
                    Data_struct.M3_vec{i,j,k} = reshape(M3_tens{i,j,k},[K*K*K 1]);
                end
            end
        end
    end
end

disp('running confusion matrix estimation algorithm');
Var_struct = alg_init(params); %Initalization
iter = 1; convergence = 0;
err_vec = zeros(max_iter,1);

Var_struct.p_vec = ones(K,1)./K;

temp_denom = 0;
temp_denom2 = 0;
Data_struct.M1_vec = M1_vec; 
Data_struct.M2_mat = M2_mat;
Var_struct.p_vec_new = Var_struct.p_vec;
for m=1:M
    temp_denom = temp_denom + params.m_vec_val(m)*norm(Data_struct.M1_vec(:,m),2)^2;
    for mm=(m+1):M
        temp_denom = temp_denom + params.M_mat_val(m,mm)*norm(Data_struct.M2_mat{m,mm},'fro')^2;
        for mmm = (mm+1):M
            temp_denom = temp_denom + params.M_tens_val(m,mm,mmm)*norm(Data_struct.M3_vec{m,mm,mmm},2)^2;
        end
    end
end
Var_struct.temp_denom = temp_denom + temp_denom2;

%Var_struct.pi_vec = params.debug.pi_vec;

Data_struct.M = M;
Gamma_iter = zeros(max_iter,1); Pi_iter = zeros(max_iter,1);

%compute objective value for the first time.
temp_obj_fun = 0; P_big = diag(Var_struct.p_vec_new);
for m=1:M
    temp_obj_fun = temp_obj_fun + params.m_vec_val(m)*0.5*norm(Data_struct.M1_vec(:,m) - Var_struct.Gamma_new{m}*Var_struct.p_vec_new,2)^2;
    for mm=(m+1):M
        temp_obj_fun = temp_obj_fun + params.M_mat_val(m,mm)*0.5*norm(Data_struct.M2_mat{m,mm} - Var_struct.Gamma_new{m}*P_big*Var_struct.Gamma_new{mm}','fro')^2;
        for mmm=(mm+1):M
            temp_obj_fun = temp_obj_fun + params.M_tens_val(m,mm,mmm)*0.5*norm(Data_struct.M3_vec{m,mm,mmm} - krp(Var_struct.Gamma_krps{mmm,mm},Var_struct.Gamma_new{m})*Var_struct.p_vec_new,2)^2;
        end
    end
end

%prev_obj_val = compute_obj_val(Data_struct,Var_struct,params);
prev_obj_val = temp_obj_fun;
params.prev_obj_val = prev_obj_val;
if ~fixedmu
    %params.inner_mu =1e-7 + 1e-2*prev_obj_val./temp_denom;
    params.inner_mu =mu_const1 + mu_const2*prev_obj_val./temp_denom;
end
if ~params.fixed_pi
    while iter < max_iter && ~convergence

        %update annotator confusion matrices and p_vec
        [Var_struct] = main_func(Data_struct,Var_struct,params);

        Gamma_iter(iter) = Var_struct.avg_Gamma_iter; Pi_iter(iter) = Var_struct.pi_iter;

        err_tmp = 0;
        for i=1:M
            %err_tmp = err_tmp + norm(Var_struct.Gamma{i} - Var_struct.Gamma_new{i},'fro')^2./norm(Var_struct.Gamma{i},'fro')^2;
            err_tmp = err_tmp + norm(Var_struct.Gamma{i} - Var_struct.Gamma_new{i},1)./norm(Var_struct.Gamma{i},1);
        end    

        temp_obj_fun = compute_obj_val(Data_struct,Var_struct,params);
        %temp_obj_fun = compute_obj_val(Var_struct);
        

        err_vec(iter) = err_tmp;
        %pverr = norm(Var_struct.p_vec - Var_struct.p_vec_new,2)^2/norm(Var_struct.p_vec,2)^2;
        pverr = norm(Var_struct.p_vec - Var_struct.p_vec_new,1)/norm(Var_struct.p_vec,1);


        if mod(iter,100) == 0 || iter == 1
            disp(['iteration - ',num2str(iter),' Gamma diff ',  num2str(err_tmp),' p_vec diff ',num2str(pverr), ' Obj_fun - ',num2str(temp_obj_fun), ' mu - ',num2str(params.inner_mu)]);
        end
        if err_tmp < tol && pverr < tol
            convergence = 1;
        end


        if ~fixedmu
            %params.inner_mu =1e-7 + 1e-2*temp_obj_fun./temp_denom;
            if temp_obj_fun < params.prev_obj_val
                params.inner_mu =mu_const1 + mu_const2*temp_obj_fun./temp_denom;
            elseif temp_obj_fun == params.prev_obj_val
                %convergence = 1;
                params.inner_mu = params.inner_mu - 1e-9*params.inner_mu;
            else
                %temp_obj_fun - params.prev_obj_val
                params.inner_mu = params.inner_mu - 1e-9*params.inner_mu;
            end
        end

        Var_struct.Gamma = Var_struct.Gamma_new;
        Var_struct.p_vec = Var_struct.p_vec_new;
        obj_fun_val(iter) = temp_obj_fun;
        params.prev_obj_val = temp_obj_fun;
        iter = iter + 1;
    end
else
    Var_struct.p_vec = params.fixed_pi_vec; Var_struct.p_vec_new = Var_struct.p_vec;
    while iter < max_iter && ~convergence

        %update annotator confusion matrices and p_vec
        [Var_struct] = main_func_fixedpi(Data_struct,Var_struct,params);

        Gamma_iter(iter) = Var_struct.avg_Gamma_iter; Pi_iter(iter) = Var_struct.pi_iter;

        err_tmp = 0;
        for i=1:M
            err_tmp = err_tmp + norm(Var_struct.Gamma{i} - Var_struct.Gamma_new{i},1)./norm(Var_struct.Gamma{i},1);
        end    
      
        temp_obj_fun = compute_obj_val_alt(Data_struct,Var_struct,params);

        err_vec(iter) = err_tmp;
        pverr = 0;
            
        if mod(iter,100) == 0 || iter == 1
            disp(['iteration - ',num2str(iter),' Gamma diff ',  num2str(err_tmp),' p_vec diff ',num2str(pverr), ' Obj_fun - ',num2str(temp_obj_fun), ' mu - ',num2str(params.inner_mu)]);
        end
        if err_tmp < tol && pverr < tol
            convergence = 1;
        end


        if ~fixedmu
            %params.inner_mu =1e-7 + 1e-2*temp_obj_fun./temp_denom;
            if temp_obj_fun < params.prev_obj_val
                params.inner_mu =mu_const1 + mu_const2*temp_obj_fun./temp_denom;
            elseif temp_obj_fun == params.prev_obj_val
                %convergence = 1;
                params.inner_mu = params.inner_mu - 1e-9*params.inner_mu;
            else
                %temp_obj_fun - params.prev_obj_val
                params.inner_mu =params.inner_mu - 1e-9*params.inner_mu;
            end
        end

        Var_struct.Gamma = Var_struct.Gamma_new;
        %Var_struct.p_vec = Var_struct.p_vec_new;
        obj_fun_val(iter) = temp_obj_fun;
        prev_obj_val = temp_obj_fun;
        iter = iter + 1;
    end
end
disp(['Algorithm finished after ',num2str(iter),' iterations. Final obj val - ',num2str(temp_obj_fun)]);
Gamma_hat = Var_struct.Gamma;
p_vec_hat = Var_struct.p_vec;

if display
    
   iter = iter - 1;
   %disp(['Objective function value = ',num2str(obj_fun)]);
   figure;
   hold all;
   plot(1:iter,err_vec(1:iter));
   %plot(1:iter,err2_vec(1:iter));
   legend('err');
   xlabel('iteration');
   ylabel('Iterate difference');
   
   figure; 
   hold all;
   plot(1:iter,obj_fun_val(1:iter))
   %legend('Objective value');
   xlabel('iteration');
   ylabel('Objective Value');
   
   figure; 
   hold all;
   plot(1:iter,Gamma_iter(1:iter));
   plot(1:iter,Pi_iter(1:iter));
   legend('Gamma','pi_vec');
   ylabel('Inner ADMM iterations');
   xlabel('BCD iteration');
end

end

