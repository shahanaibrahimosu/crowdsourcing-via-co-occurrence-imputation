function [A_new] = N_CTF_AO_KL_sub(Y_p,I,F,A,lambda,marg,rows,cols,n,max_iter,tol)

% Armijo rule step size!
beta  = 0.9;  % usuallt between 1/2 and 1/10
sigma = 1e-3; % usually it is between 10^-5 and 10^-1
step  = 0.1;%0.5;  % usually 1

%%%%%%%%%%%%%%%%%%%%%%%%
Y_pn = cell(length(Y_p),1);
for i=1:length(Y_p)
    nz{i} = find(Y_p{i}(:));
    Y_pn{i} = Y_p{i}(nz{i});
end

Q = cell(length(rows),1);
for i=1:length(rows)
    ps   = cols(i);
    if (length(marg{rows(i)})>1)
        Q{i} = kr(A(marg{rows(i)}([end:-1:ps+1 ps-1:-1:1])))*diag(lambda);
    else
        Q{i} = lambda';
    end
end
c = [];
A_new = A{n};

cost_prev = cost(A_new,Y_p,Q);
for iter = 1:max_iter
    grad_A = zeros(I(n),F);
    for i=1:length(rows)
        d           = A_new*Q{i}';
        d           = max(d,10^-6);
        d=d./sum(d,"all");
        frac        = zeros(size(d));
        frac(Y_p{i}(:)~=0) = Y_pn{i}./d(Y_p{i}(:)~=0);
        grad_A      = grad_A - frac*Q{i};
    end
    grad_A;
    %grad_A = grad_A./length(rows);
%     grad_A(abs(grad_A)>7)=0;
%     grad_A(isinf(grad_A))=0;
%     grad_A(isnan(grad_A))=0;
    grad_A;
    A_old  = A_new;
    [A_new,cost_new]  = armijo_rule(A_old,Y_p,Q,grad_A,beta,sigma,step,cost_prev,nz);
    cost_new;
    c         = [c; cost_new];
    cost_prev = c(end);
    
    
    % semilogy(c)
    % drawnow
    if (norm(A_old(:) - A_new(:))/norm(A_old(:)) < tol)
        break;
    end
end
c;
end

function [A,cost_new] = armijo_rule(A_old,Y_p,Q,grad_A,beta,sigma,step,cost_prev,nz)
A_new     = A_old .* exp(-step*grad_A);
A_new     = bsxfun(@times,A_new,1./sum(A_new,1));
cost_new  = cost(A_new,Y_p,Q);
d = A_old - A_new;

while sum(isnan(A_new(:)))>0 || (cost_prev-cost_new) < sigma * grad_A(:)'*d(:)
    step      = step*beta;
    A_new     = A_old .* exp(-step*grad_A);
    A_new     = bsxfun(@times,A_new,1./sum(A_new,1));
    cost_new  = cost(A_new,Y_p,Q);
    d = A_old - A_new;
    if step<1e-5
        A_new = A_old;
        break
    end
end
A = A_new;
end

function c = cost(A,Y_p,Q)
c = 0;
for i=1:length(Q)
    d       = A*Q{i}';
    d           = max(d,10^-6);
    d=d./sum(d,"all");
    frac        = zeros(size(d));
    frac(Y_p{i}(:)~=0) = Y_p{i}(Y_p{i}(:)~=0)./d(Y_p{i}(:)~=0);
    %frac = Y_p{i}./d;
    temp    = Y_p{i}(Y_p{i}(:)~=0).*log(frac(Y_p{i}(:)~=0));
    %temp = Y_p{i}(:).*log(frac(:));
    temp(isnan(temp)) = 0;
    c = c + sum(temp);
end
end