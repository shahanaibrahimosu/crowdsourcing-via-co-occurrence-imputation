function dist_s = getSubDistance(M,U)
M = diag(1./vecnorm(M',2))*M;
U = diag(1./vecnorm(U',2))*U;
N = size(U,2);
P_m = (eye(N)-M'*pinv(M*M')*M);
Q_u = orth(U');
[~,dist_s,~] = svds(P_m*Q_u,1);

end

