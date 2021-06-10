function [Var_struct] = main_func_fixedpi(Data_struct,Var_struct,params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%update annotator confusion matrices
M = Data_struct.M;
avg_Gamma_iter = 0;

Var_struct.pi_iter = 0;
indx = 1:M;
for i=1:M
    %Var_struct.Gamma_new = Var_struct.Gamma;
    M3_tmp = squeeze(Data_struct.M3_mat(i,:,:));
    [Var_struct.Gamma_new{i},Var_struct.Delta{i},iter] = inner_ADMM(Data_struct.M1_vec(:,i),Data_struct.M2_mat,M3_tmp,Var_struct.Gamma_prods,Var_struct.Gamma_krps,Var_struct.p_vec,Var_struct.Gamma_new,Var_struct.Delta{i},i,params);

    Var_struct.Gamma_prods{i} = Var_struct.Gamma_new{i}'*Var_struct.Gamma_new{i}; %update cached self-products.
    
    for j = indx(1:(i-1)) %update cached khatri-rao products.
        Var_struct.Gamma_krps{i,j} = krp(Var_struct.Gamma_new{i},Var_struct.Gamma_new{j});
        Var_struct.Gamma_krps{j,i} = krp(Var_struct.Gamma_new{j},Var_struct.Gamma_new{i});
    end
    for j = indx((i+1):end)
        Var_struct.Gamma_krps{i,j} = krp(Var_struct.Gamma_new{i},Var_struct.Gamma{j});
        Var_struct.Gamma_krps{j,i} = krp(Var_struct.Gamma{j},Var_struct.Gamma_new{i});
    end

    avg_Gamma_iter = avg_Gamma_iter + iter;
end
%[Var_struct.p_vec_new,Var_struct.delta,Var_struct.pi_iter,Var_struct.lhsmat,Var_struct.rhsmat] = inner_ADMM_p_upd(Data_struct.M1_vec,Data_struct.M2_mat,Data_struct.M3_vec,Var_struct.Gamma_prods,Var_struct.Gamma_krps,Var_struct.p_vec,Var_struct.Gamma_new,Var_struct.delta,params);

%Var_struct.Gamma_new = params.debug.Gamma;
Var_struct.avg_Gamma_iter = avg_Gamma_iter./M;

end



