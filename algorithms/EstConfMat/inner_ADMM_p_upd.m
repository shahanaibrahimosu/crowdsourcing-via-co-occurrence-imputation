function [p_vec_i,delta_i,iter,lhsmat,rhsmat] = inner_ADMM_p_upd(M1_vec,M2_mat,M3_vec,Gamma_prods,Gamma_krps,p_vec,Gamma,delta,params)
%INNER_ADMM Summary of this function goes here
%   Detailed explanation goes here

tol = params.inner_tol;
max_iter = params.inner_max_iter_pi;
mu = params.inner_mu;
%lambda = params.inner_lambda_pi;

M = size(M2_mat,2);
K = length(p_vec);

Tmp_mat = zeros(K,K);
Tmp_mat2 = zeros(K,1);
term_num = 0;
%Delta = zeros(K,K);
for i=1:M
       Tmp_mat = Tmp_mat + params.m_vec_val(i)*Gamma_prods{i};
       Tmp_mat2 = Tmp_mat2 + params.m_vec_val(i)*Gamma{i}'*M1_vec(:,i);
       for j=(i+1):M
               tmpprod = Gamma_prods{j}.*Gamma_prods{i}; 
               Tmp_mat = Tmp_mat + params.M_mat_val(i,j)*(tmpprod);
               %Tmp_mat = Tmp_mat + params.M_mat_val(i,j)*(Gamma_krps{j,i}'*Gamma_krps{j,i});
			   Tmp_mat2 = Tmp_mat2 + params.M_mat_val(i,j)*Gamma_krps{j,i}'*M2_mat{i,j}(:);
               for k = (j+1):M
                  term_num = term_num + 1;
                  tmp_iter_mat = krp(Gamma_krps{k,j},Gamma{i});           
                  %Tmp_mat = Tmp_mat + params.M_tens_val(i,j,k)*(tmp_iter_mat'*tmp_iter_mat);
                  Tmp_mat = Tmp_mat + params.M_tens_val(i,j,k)*(Gamma_prods{k}.*Gamma_prods{j}.*Gamma_prods{i});
                  Tmp_mat2 = Tmp_mat2 + params.M_tens_val(i,j,k)*tmp_iter_mat'*M3_vec{i,j,k};   
               end
       end
end


lambda = (norm(Tmp_mat,'fro')^2)/(term_num*K);
lambda = (norm(Tmp_mat,'fro'))/(K);
%lambda = (norm(Tmp_mat,'fro')^2)/(K);

lhsmat = Tmp_mat;
Tmp_mat = Tmp_mat + (mu + lambda)*eye(K);

L = chol(Tmp_mat,'lower'); %inversion using cholesky factorization.

rhsmat = Tmp_mat2;
Tmp_mat2 = Tmp_mat2 +  mu*p_vec;

u_init = p_vec;
p_vec_i = p_vec;

iter = 1; convergence = 0;
while iter <= max_iter && ~convergence
   u_new = L'\(L\(Tmp_mat2 + lambda*(p_vec_i + delta)));
   p_vec_new = ProjectOntoSimplex1(u_new - delta,1,[K 1]);
   delta = delta + p_vec_new - u_new;
   
   err1 = norm(p_vec_new - u_new,2)^2/norm(p_vec_new,2)^2; 
   err2 = norm(p_vec_new - u_init,2)^2/norm(delta,2)^2;
   if err1 < tol && err2 < tol 
       convergence = 1;
   end
   p_vec_i = p_vec_new; 
   iter = iter + 1;
end

delta_i = delta;
end

