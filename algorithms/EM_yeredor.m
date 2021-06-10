function [mu,d,timestamps] = EM_yeredor(mu,d,f,opts)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = length(f);
k = size(mu{1},1);
m = length(mu);
[A] = convert_for_comp(f);
Z = zeros(n,k,m);
for i = 1:size(A,1)
 Z(A(i,1),A(i,3),A(i,2)) = 1;
end

% n=size(Z{1},1);
% k=size(mu{1},2);
% m=length(Z);

Nround=opts.Nround;
% EM update
%[rec_error(1),~,~] = getMSE_nomre(mu,d,opts.A_true,opts.l_true);
a=tic;
for iter = 1:Nround
    q = zeros(n,k);
    q1 = zeros(k,k);
    for i=1:m
        tmp = mu{i};
        tmp(find(tmp ==0)) = eps;
        tmp(find(isnan(tmp))) = eps;
        tmp1 = d;
        tmp1(find(tmp1 ==0)) = eps;
        tmp1(find(isnan(tmp1))) = eps;
        q = q + Z(:,:,i)*log(tmp);
        %q = q + Z(:,:,i)*log(mu(:,:,i));
    end
    q=q+repmat(log(tmp1'),n,1);
    q = exp(q);
    q = bsxfun(@rdivide,q,sum(q,2));
    


    for i = 1:m
        mu{i} = (Z(:,:,i))'*q;
        
        mu{i} = AggregateCFG(mu{i},0);
        tmp_vec = sum(mu{i});
        indx = find(tmp_vec > 0);
        mu{i}(:,indx) = bsxfun(@rdivide,mu{i}(:,indx),tmp_vec(indx));
    end
    d = sum(q,1)';
    tmp_vec = sum(d);
    indx = find(tmp_vec > 0);
    d = bsxfun(@rdivide,d,tmp_vec);
    b=toc(a);
    timestamps(iter)=b;
%     [rec_error(iter+1),~,~] = getMSE_nomre(mu,d,opts.A_true,opts.l_true);
%     if(abs(rec_error(iter+1)-rec_error(iter))/rec_error(iter) < opts.tol)
%       break;
%     end
    a=tic;
end

%error_EM_predict = mean(y(valid_index) ~= (J(valid_index))');
%EM_err = error_EM_predict(end);

end

