function X = impute_block(M_mat,m,n,l_ref1,l_ref2)

K = size(M_mat{m,l_ref2},1);
%% Get annottaors who has with nonempty satistics with an annottaor i
C = [M_mat{m,l_ref2}' M_mat{l_ref1,l_ref2}']';
D = [M_mat{n,l_ref1} M_mat{n,l_ref2}];

[U,S,V] = svds(C,K);
U1 = U(1:K,:);
U2= U(K+1:end,:);
[U,S,V] = svds(D,K);
V1 = V(1:K,:);
V2 = V(K+1:end,:);
X = U1*pinv(U2)*V1*pinv(V2)*M_mat{n,l_ref2}';
X = abs(X);
X = X./sum(X,'all');
X(isnan(X))=0;

end