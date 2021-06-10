function [H_bar,Q] = alg_SymNMF(U,opts)

%% Getting parameters
M= opts.M;
A = cell(M,1);
K = size(U,2);
% Q_true=opts.Q_true;
Q=opts.Q_init;
% H_true=opts.H_true;
for iter=1:opts.n_iter
    %% Updating H    
    H_bar = max(1e-6,U*Q);
    
%     %% Updating A and D
%     H=H_bar';
%     H_t=[];
%     for i=1:M
%         A{i} = transpose(H(:,(i-1)*K+1:i*K));
%         A{i} = max(eps,A{i});
%         A{i} = bsxfun( @times, A{i}, 1./sum(A{i}) );
%         H_t = [H_t A{i}'];
%         %H(:,(i-1)*K+1:i*K) = (sqrt(D_lambda))*A{i}';
%     end 
% 
%     D_lambda =(H*pinv(H_t)).^2;
%     %
%     H  = sqrt(D_lambda)*H_t;
%     H_bar=H';
    
    %% Updating Q
    [U_t,D,V_t]=svds(H_bar'*U,K);
    Q = V_t*U_t';    
    
    %% checking
    abs(trace(H_bar'*(H_bar-U*Q)));
    if(abs(trace(H_bar'*(H_bar-U*Q))) <opts.tol)
         break;
    end   
end
% lambda = diag(D_lambda);
end