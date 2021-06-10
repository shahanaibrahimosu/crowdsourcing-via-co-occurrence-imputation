function X = impute_block_mod(M_mat,m,n,l_ref1,l_ref2,K)

% K = size(M_mat{m,l_ref2},1);
%% Get annottaors who has with nonempty satistics with an annottaor i
C = [M_mat{m,l_ref1}' M_mat{l_ref2,l_ref1}']';
D = [M_mat{n,l_ref2}];

c_column_check = sum(sum(C,2)~=0);
d_column_check = sum(sum(D,1)~=0);

if(c_column_check==2*K && d_column_check==K)
    [U,S1,V] = svds(C,K);
    U1 = U(1:K,:);
    U2= U(K+1:end,:);
%     [Un,S2,V] = svds(D,K);
%     V1 = V(1:K,:);
    % V2 = V(K+1:end,:);
    X = U1*pinv(U2)*D';
    X = abs(X);
    X = X./sum(X,'all');
    X(isnan(X))=0;
else
    X = zeros(K,K);
end

if(rank(X)~=K)
    X= zeros(K,K);
end
X(isnan(X))=0;
X(isinf(X))=0;


end