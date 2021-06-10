function [EM_err,A] = run_EM_Spectral(f,k,y,Nround)
%===================== EM with spectral method ==============
% method of moment

n = length(f);
m = size(f,1);
[A] = convert_for_comp(f);
Z = zeros(n,k,m);
for i = 1:size(A,1)
 Z(A(i,1),A(i,3),A(i,2)) = 1;
end

valid_index = find(y >0);
mode = 0;
[n,k,m] = size(Z);
group = mod(1:m,3)+1;
Zg = zeros(n,k,3);
cfg = zeros(k,k,3);
for i = 1:3
    I = find(group == i);
    Zg(:,:,i) = sum(Z(:,:,I),3);
end

x1 = Zg(:,:,1)';
x2 = Zg(:,:,2)';
x3 = Zg(:,:,3)';

muWg = zeros(k,k+1,3);
muWg(:,:,1) = SolveCFG(x2,x3,x1);
muWg(:,:,2) = SolveCFG(x3,x1,x2);
muWg(:,:,3) = SolveCFG(x1,x2,x3);

mu = zeros(k,k,m);
for i = 1:m
    x = Z(:,:,i)';
    x_alt = sum(Zg,3)' - Zg(:,:,group(i))';
    muW_alt = (sum(muWg,3) - muWg(:,:,group(i)));
    mu(:,:,i) = (x*x_alt'/n) / (diag(muW_alt(:,k+1)/2)*muW_alt(:,1:k)');
    
    mu(:,:,i) = max( mu(:,:,i), 10^-6 );
    mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
    for j = 1:k
        mu(:,j,i) = mu(:,j,i) / sum(mu(:,j,i));
    end
end

% EM update
for iter = 1:Nround
    q = zeros(n,k);
    for j = 1:n
        for c = 1:k
            for i = 1:m
                if Z(j,:,i)*mu(:,c,i) > 0
                    q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
                end
            end
        end
        q(j,:) = exp(q(j,:));
        q(j,:) = q(j,:) / sum(q(j,:));
    end

    for i = 1:m
        mu(:,:,i) = (Z(:,:,i))'*q;
        
        mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
        for c = 1:k
            mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
        end
    end
end

[I J] = max(q');
error2_predict = mean(y(valid_index) ~= (J(valid_index))');
EM_err = error2_predict(end);

A =cell(m,1);
for i=1:m
    A{i}=mu(:,:,i);
end
end