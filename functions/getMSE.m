function [MSE] = getMSE(A_est,A_true, l_est,l_true)
A_est_cat   = concatenate(A_est);
A_true_cat  = concatenate(A_true);
N = length(A_true);
F = length(l_true);
Pm  = Hungarian(-real(A_est_cat)'*real(A_true_cat)); 
MSE=0;
% Compute error
for n=1:N
    A_est{n} = A_est{n}*Pm;
    error = norm(A_est{n}(:) - A_true{n}(:))^2;
    MSE =  MSE + error;
end

l_est = Pm*l_est;
MSE_l = norm(l_est-l_true)^2;
MSE= (MSE+MSE_l)/(N*F+1);

end
