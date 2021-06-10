function [lambda_new,out] = N_CTF_AO_KL_sub_lambda(Y,F,A,lambda,marg,max_iter,tol)

% Armijo rule
beta  = 0.9;
sigma = 1e-3;
step  = 0.1;
Y_n = cell(length(Y),1);
for i=1:length(Y)
    nz{i} = find(Y{i}(:));
    Y_n{i} = Y{i}(nz{i});
end

Q = cell(length(marg),1);
for i=1:length(marg)
    if (length(marg{i})>1)
        Q{i} = kr(A(marg{i}(end:-1:1)));
    else
        Q{i} = A{marg{i}};
    end
end

c = [];
lambda_new = lambda;
cost_prev = cost(lambda_new,Y,Q,nz);

c=cost_prev;
for iter = 1:max_iter
    grad_lambda = zeros(F,1);
    
    for i=1:length(marg)
        d    = Q{i}*lambda_new;
        d           = max(d,10^-6);
        %d    = d+10e-6;
        d=d./sum(d,"all");
        frac = zeros(size(d));
        frac(Y{i}(:)~=0) = Y{i}(Y{i}(:)~=0)./d(Y{i}(:)~=0);
        grad_lambda = grad_lambda - Q{i}'*frac;
    end
%     grad_lambda(abs(grad_lambda)>7)=0;
%     grad_lambda(isinf(grad_lambda))=0;
%     grad_lambda(isnan(grad_lambda))=0;    
    lambda_old  = lambda_new;
    [lambda_new,cost_new]  = armijo_rule(lambda_new,Y,Q,grad_lambda,beta,sigma,step,cost_prev,nz);
    c = [c; cost_new];
    cost_prev = c(end);
        
    if (norm(lambda_old(:) - lambda_new(:))/norm(lambda_old(:)) < tol)
        break;
    end
end
out.cost = c(end);
%out.cost=0;
end

function [lambda,cost_new] = armijo_rule(lambda_old,Y,Q,grad_lambda,beta,sigma,step,cost_prev,nz)
lambda_new = lambda_old .* exp(-step*grad_lambda);
lambda_new = lambda_new/sum(lambda_new );
cost_new   = cost(lambda_new,Y,Q,nz);
d          = lambda_old - lambda_new;

while  sum(isnan(lambda_new(:)))>0 || (cost_prev-cost_new) < sigma * grad_lambda(:)'*d(:)
    step = step*beta;
    lambda_new = lambda_old .* exp(-step*grad_lambda);
    lambda_new = lambda_new/sum(lambda_new);
    cost_new  = cost(lambda_new,Y,Q,nz);
    d = lambda_old - lambda_new;
    
    if step<1e-5
        lambda_new = lambda_old;
        break
    end
end
lambda = lambda_new;
end

function c = cost(lambda,Y,Q,nz)
c = 0;
for i=1:length(Q)
    d    = Q{i}*lambda;
    d           = max(d,10^-6);
    %d    = d+10e-6;
    d=d./sum(d,"all");
    frac = Y{i}(Y{i}(:)~=0)./d(Y{i}(:)~=0);
    temp    = Y{i}(Y{i}(:)~=0).*log(frac(:));
    temp(isnan(temp)) = 0;
    c = c + sum(temp);
end
c=c./length(Q);
end