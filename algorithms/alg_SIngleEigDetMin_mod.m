function[A,lambda,time_total] =  alg_SIngleEigDetMin_mod(M_mat,K,ops)
%% Get parameters
M = length(M_mat);
% A_true = ops.A_true;
%% Initializing U and A
U = cell(M,1);
A = cell(M,1);

if(ops.eigen_decomposition || ops.cf_init_via_eigendecomp)
    a=tic;
    %% Impute the missing statistics
    if(ops.impute_method)
        opts={};
        opts.m_star_1=ops.m_star_1;
        opts.m_star_2=ops.m_star_2;
        opts.impute_method=ops.impute_method;
        opts.percent_blocks_to_impute=ops.percent_blocks_to_impute;
        opts.N_mat_valid = ops.N_mat_valid;
        opts.M_mat_true=ops.M_mat_true;
        opts.no_blocks_imputation=ops.no_blocks_imputation;
        M_mat = alg_impute_blocks(M_mat,K,opts);
    end
    

    %% Construct Big Matrix with all statistics
    X=[];
    for i=1:M
        X1=[];
        for j=1:M
            if(isempty(M_mat{i,j}))
                M_mat{i,j} = zeros(K,K);
            end  
            X1 = [X1 M_mat{i,j}];
        end
        X = [X; X1];
    end

    %% Perform eigen decomposition
    %[U_r,D]=eigs(X,K);
    [U_r,D] = svds(X,K);
    U_r = U_r*sqrt(D);

    U_r=U_r';

    for i=1:M
        U{i}=U_r(:,(i-1)*K+1:i*K)';
    end
    time_subspace= toc(a);
end
if(ops.eigen_decomposition==0)
%% Improve the subspace estimation
    ops1={};
    ops1.lambda=1e-6;
    ops1.max_iter=20;
    ops1.tol=1e-6;
    ops1.p=2;
    ops1.epsilon=1e-6;
    [U,time_subspace] = coupled_fact_U_constrained_mod(M_mat,U,K,ops1);
    U_r=[];
    for i=1:M
        U_r=[U_r U{i}'];
    end
end   


% H_true=[];
% for i=1:M
%     H_true=[H_true sqrt(diag(ops.l_true))*A_true{i}'];
% end
% dist=getSubDistance(U_r,H_true);
% lambda_prior=ones(K,1)/K;
% D_lambda= diag(lambda_prior);
% Q_true = U_r*pinv(H_true);



%% Solve H 
ops1={};
ops1.Q_init=eye(K);
% ops1.Q_init=orth(ops1.Q_init);
% ops1.Q_true=Q_true;
% ops1.H_true=H_true;
ops1.n_iter=200;
ops1.tol = 1e-14;
ops1.M=M;
a=tic;
[H_bar,Q]=alg_SymNMF_mod(U_r',ops1);

H= H_bar';

%% estimating A
H_t=[];
for i=1:M
    A{i} = transpose(H(:,(i-1)*K+1:i*K));
    A{i} = max(eps,A{i});
    A{i} = bsxfun( @times, A{i}, 1./sum(A{i}) );
    H_t = [H_t A{i}'];
    %H(:,(i-1)*K+1:i*K) = (sqrt(D_lambda))*A{i}';
end 

%% estimating lambda
D_lambda =(H*pinv(H_t)).^2;
lambda = diag(D_lambda);
A_est_mod={};
c=1;
for i=1:length(A)
    G =A{i};
    G = max( G, 1e-3 );
    t=sum(G,1);
    G = G*diag(1./t);
    A{i}=G;
    if(rank(A{i})==K)
        A_est_mod{c}=A{i};
        c=c+1;
    end
end

A=A_est_mod;

time_symnmf=toc(a);
time_total= time_symnmf+time_subspace;

end