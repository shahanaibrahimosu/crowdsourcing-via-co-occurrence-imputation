function [W,H,A,lambda] = alg_alternating_optim_withD(U,opts)

%% Getting parameters
K = size(U,1);
M = opts.M;
H = opts.H_init;
H_prev=H;
W_true=opts.W_true;
H_true=opts.H_true;
l_prior=opts.lambda_prior;
D_lambda=diag(l_prior);
A=cell(M,1);
for iter=1:opts.n_iter
    %% Updating W
    [U_t,D,V_t]=svds(U*pinv(H),K);
    W = U_t*V_t';
    %% Updating H
    opts1.max_iter=20;
    opts1.tol=1e-6;
    opts1.W = W;
    opts1.H_init=H;
    opts1.W_true = W_true;
    opts1.H_true = H_true;
    H = ProjGD(U,opts1);
    
    % projection & estimating D
    H_bar=[];
    for i=1:M
        A{i} = transpose(H(:,(i-1)*K+1:i*K));
        A{i} = max(eps,A{i});
        A{i} = bsxfun( @times, A{i}, 1./sum(A{i}) );
        H_bar = [H_bar A{i}'];
        %H(:,(i-1)*K+1:i*K) = (sqrt(D_lambda))*A{i}';
    end 
    D_lambda =(H*pinv(H_bar)).^2;
    
    if((norm(H_prev-H)/length(U))<opts.tol)
         break;
    end   
    H_prev=H;
end
lambda = diag(D_lambda);
end



