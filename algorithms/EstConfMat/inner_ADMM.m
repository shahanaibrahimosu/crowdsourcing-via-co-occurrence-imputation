function [Gamma_i,Delta_i,iter] = inner_ADMM(M1_vec,M2_mat,M3_mat,Gamma_prods,Gamma_krps,p_vec,Gamma,Delta,i,params)
%INNER_ADMM Summary of this function goes here
%   Detailed explanation goes here

tol = params.inner_tol;
max_iter = params.inner_max_iter;
mu = params.inner_mu;

M = size(M2_mat,2);
K = length(p_vec);
P = diag(p_vec);

Tmp_mat = zeros(K,K);
Tmp_mat2 = zeros(K,K);
term_num = 0;

for j=1:M
    if i~=j
       %term_num = term_num + 1;
       Tmp_mat = Tmp_mat + params.M_mat_val(i,j)*Gamma_prods{j};
       Tmp_mat2 = Tmp_mat2 + params.M_mat_val(i,j)*Gamma{j}'*M2_mat{j,i};
       for k=(j+1):M
           if k~=i
               term_num = term_num + 1;
               Tmp_mat = Tmp_mat + params.M_tens_val(i,j,k)*Gamma_prods{k}.*Gamma_prods{j};
               Tmp_mat2 = Tmp_mat2 + params.M_tens_val(i,j,k)*Gamma_krps{k,j}'*M3_mat{j,k};
           end
       end
    end
end

Tmp_mat = P*Tmp_mat*P + params.m_vec_val(i)*(p_vec*p_vec');

lambda = (norm(Tmp_mat,'fro')^2)/(term_num*K);
lambda = (norm(Tmp_mat,'fro'))/(K);
%lambda = (norm(Tmp_mat,'fro')^2)/(K);


Tmp_mat = Tmp_mat + (mu + lambda)*eye(K);

L = chol(Tmp_mat,'lower'); %invert using cholesky factorization.

Tmp_mat2 = P*Tmp_mat2 + params.m_vec_val(i)*p_vec*M1_vec' + mu*Gamma{i}';



Gamma_init = Gamma{i};
Gamma_i = Gamma{i};
iter = 1; convergence = 0;
while iter <= max_iter && ~convergence
   Phi_new = L'\ (L\(Tmp_mat2 + lambda*(Gamma_i + Delta)'));
   tempmat = Phi_new' - Delta;

   Gamma_i_new = ProjectOntoSimplex1(tempmat,1,[K K]);
   Delta = Delta + Gamma_i_new - Phi_new';
   
   err1 = norm(Gamma_i_new - Phi_new','fro')^2/norm(Gamma_i_new,'fro')^2; err2 = norm(Gamma_i_new - Gamma_init,'fro')^2/norm(Delta,'fro')^2;
   if err1 < tol && err2 < tol 
       convergence = 1;
   end
   Gamma_i = Gamma_i_new; 
   iter = iter + 1;
end

Delta_i = Delta;
end

