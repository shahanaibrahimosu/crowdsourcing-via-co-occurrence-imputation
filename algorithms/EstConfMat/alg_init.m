function [Var_struct] = alg_init(params)
%ALG_INIT Function to initialize EstConfMat_AO_ADMM
% Panagiotis Traganitis - email: traga003@umn.edu
M = params.M; K = params.K;

disp('Initializing');
%%%%%%%%%% initialization %%%%%%%%%%%%%%%%%%%%
Var_struct.Gamma = cell(M,1);
Var_struct.Gamma_new = cell(M,1);
Var_struct.Delta = cell(M,1);
if params.init == 1
    for i=1:M
       Var_struct.Gamma{i} = rand(K,K); %create a random KxK matrix with entries in [0,1]
       Var_struct.Gamma{i} = bsxfun(@rdivide,Var_struct.Gamma{i},sum(Var_struct.Gamma{i})); %normalize its columns so that they sum up to 1
       [vl,indx] = max(Var_struct.Gamma{i}); %move maximum value to diagonal
        vl2 = diag(Var_struct.Gamma{i}); 
        Var_struct.Gamma{i} = Var_struct.Gamma{i} - diag(vl2) + diag(vl);
        for j=1:K
            Var_struct.Gamma{i}(indx(j),j) = vl(j);
        end
       %Gamma{i} = bsxfun(@rdivide,Gamma{i},sum(Gamma{i}));
       Var_struct.Delta{i} = zeros(K,K); 
       Var_struct.Gamma_new{i} = Var_struct.Gamma{i};
    end
    Var_struct.p_vec = rand(K,1); Var_struct.p_vec = Var_struct.p_vec./sum(Var_struct.p_vec);

elseif params.init == 2
    disp('Initializing with majority voting');
    idx_MV = params.idx_MV;
    N = numel(idx_MV);
    f = params.f; 
    for m=1:M
       Var_struct.Gamma{m} = zeros(K,K);
    end
    Var_struct.p_vec = zeros(K,1);
    for k=1:K
        indx = idx_MV == k;
        for m=1:M
            f_tmp = f(m,indx);
            for kk = 1:K
                Var_struct.Gamma{m}(kk,k) = sum(f_tmp == kk)/sum(indx);
            end
        end
        Var_struct.p_vec(k) = sum(indx)/N;
    end
    for m=1:M
       Var_struct.Delta{m} = zeros(K,K); 
       Var_struct.Gamma_new{m} = Var_struct.Gamma{m}; 
    end
else
    for i=1:M
       Var_struct.Gamma{i} = rand(K,K);
       Var_struct.Gamma{i} = bsxfun(@rdivide,Var_struct.Gamma{i},sum(Var_struct.Gamma{i}));
       Var_struct.Delta{i} = zeros(K,K); 
       Var_struct.Gamma_new{i} = Var_struct.Gamma{i};
    end
    Var_struct.p_vec = rand(K,1); Var_struct.p_vec = Var_struct.p_vec./sum(Var_struct.p_vec);

end
%p_vec = (1/K)*ones(K,1);
Var_struct.delta = zeros(K,1);

%Var_struct.Gamma = params.debug.Gamma;

Var_struct.Gamma_prods = cell(M,1);
Var_struct.Gamma_krps = cell(M,M);
for i=1:M
   Var_struct.Gamma_prods{i} = Var_struct.Gamma{i}'*Var_struct.Gamma{i}; 
   for j=1:M
      if j~=i
         Var_struct.Gamma_krps{i,j} = krp(Var_struct.Gamma{i},Var_struct.Gamma{j}); 
      end
   end
end


end

