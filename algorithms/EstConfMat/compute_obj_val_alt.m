function [obj_fun_val] = compute_obj_val_alt(Data_struct,Var_struct,params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Take advantage of the stuff that has been precomputed during the pi_vec
%update.
%temp_obj_fun = 0.5*Var_struct.temp_denom - Var_struct.p_vec_new'*Var_struct.rhsmat + 0.5*Var_struct.p_vec_new'*Var_struct.lhsmat*Var_struct.p_vec_new;

%obj_fun_val = temp_obj_fun;


 M = Data_struct.M;
 
 temp_obj_fun = 0; P_big = diag(Var_struct.p_vec_new);
 for m=1:M
     temp_obj_fun = temp_obj_fun + params.m_vec_val(m)*0.5*norm(Data_struct.M1_vec(:,m) - Var_struct.Gamma_new{m}*Var_struct.p_vec_new,2)^2;
     for mm=(m+1):M
         temp_obj_fun = temp_obj_fun + params.M_mat_val(m,mm)*0.5*norm(Data_struct.M2_mat{m,mm} - Var_struct.Gamma_new{m}*P_big*Var_struct.Gamma_new{mm}','fro')^2;
         for mmm=(mm+1):M
             temp_obj_fun = temp_obj_fun + params.M_tens_val(m,mm,mmm)*0.5*norm(Data_struct.M3_vec{m,mm,mmm} - krp(Var_struct.Gamma_krps{mmm,mm},Var_struct.Gamma_new{m})*Var_struct.p_vec_new,2)^2;
         end
     end
 end
 
 obj_fun_val = temp_obj_fun;

end