function [M_mat,N_valid_mat,M_tens,mean_vec,M_tens_val,M_mat_val,N_valid_tens] = calc_annotator_moments(F,moments)
%CALC_ANNOTATOR_MOMENTS function to compute the first and second (cross covariance) moments of
%between the annotators responses.
%   Input: F - Mx1 cell containing KxN matrices of annotator responses
%    moments - vector indicating which moments to compute i.e. to compute
%    1st and 2nd order moments then moments = [1;2]; Up to 3rd order
%    moments
%  Output:M_tens - MxMxM cell containing KxKxK crosscorrelation tensors 
%          M_mat - MxM cell containing KxK crosscorrelation matrices.
%       mean_vec - KxM matrix, where the i-th column is the mean of F{i}
%         M_mat_val,M_valid_tens - values that indicate whether some crosscorrelations are
%         all 0 or not. 
%   Panagiotis Traganitis. traga003@umn.edu
if nargin < 2
    moments = [1;2;3];
end


M = length(F); [K,N] = size(F{1});

mean_vec = zeros(K,M);
Tmp_mats = cell(M,1);
Tmp_mats_transp = cell(M,1);
%Tidx_r = cell(M,1); Tidx_c = cell(M,1);
Tidx = cell(M,K);
tmpinx = cell(M,1);
tmpinx2 = cell(M,M);
M_mat = cell(M,M);
M_mat_val = zeros(M,M);
N_valid_mat = zeros(M,M);
M_tens = cell(M,M,M);
M_tens_val = zeros(M,M,M);
N_valid_tens = zeros(M,M,M);


for i=1:M
     %compute mean of F{i}
    Tmp_mats{i} = full(F{i});
    tmpvec = sum(Tmp_mats{i});
    tmpinx{i} = tmpvec ~= 0;
    N_valid = sum(tmpinx{i});
    %mean_vec(:,i) = mean(Tmp_mats{i},2)./p_obs(i);
    mean_vec(:,i) = sum(Tmp_mats{i},2)./N_valid;
    Tmp_mats_transp{i} = Tmp_mats{i}';
    %[Tidx_r{i},Tidx_c{i}] = find(Tmp_mats{i});
    [Tidx_r,Tidx_c] = find(Tmp_mats{i});
    for kk=1:K
       Tidx{i,kk} = Tidx_c(Tidx_r == kk); 
    end
end

if ~ismember(1,moments)
    mean_vec = zeros(K,M);
else
    disp('Computing 1st order moments');
end
    if ismember(2,moments) || ismember(3,moments)
        disp('Computing 2nd order moments');
        for i=1:M
            for j=(i+1):M
                %inx = intersect(tmpinx{i},tmpinx{j}); 
                tmpinx2{i,j} = tmpinx{i}.*tmpinx{j};
                N_valid = sum(tmpinx2{i,j}); 
                N_valid_mat(i,j) = N_valid; N_valid_mat(j,i) = N_valid;
                if N_valid == 0
                     N_valid = 1;
                     M_mat_val(i,j) = 0; 
                     M_mat_val(j,i) = 0;
                else
                    M_mat_val(i,j) = 1;
                    M_mat_val(j,i) = M_mat_val(i,j);
                end
                M_mat{i,j} = (1/(N_valid)).*full(Tmp_mats{i}*Tmp_mats_transp{j}); %Compute crosscorrelations
                M_mat{j,i} = M_mat{i,j}';
            end
        end
    end

    if ismember(3,moments)
        disp('Computing 3rd order moments');
        for i=1:M
           for j=(i+1):M
                  for k=(j+1):M
                         Tmp = zeros(K,K,K);
                         inx1 = tmpinx2{i,j}.*tmpinx{k};
                         N_valid = sum(inx1); 
                         N_valid_tens(i,j,k) = N_valid;
                         N_valid_tens(i,k,j) = N_valid;
                         N_valid_tens(j,i,k) = N_valid;
                         N_valid_tens(j,k,i) = N_valid;
                         N_valid_tens(k,i,j) = N_valid;
                         N_valid_tens(k,j,i) = N_valid;
                         if N_valid == 0
                             N_valid = 1;
                             M_tens_val(i,j,k) = 0;
                             M_tens_val(i,k,j) = 0;
                             M_tens_val(j,i,k) = 0;
                             M_tens_val(j,k,i) = 0;
                             M_tens_val(k,i,j) = 0;
                             M_tens_val(k,j,i) = 0;
                         else
                             M_tens_val(i,j,k) = 1;
                             M_tens_val(i,k,j) = M_tens_val(i,j,k);
                             M_tens_val(j,i,k) = M_tens_val(i,j,k);
                             M_tens_val(j,k,i) = M_tens_val(i,j,k);
                             M_tens_val(k,i,j) = M_tens_val(i,j,k);
                             M_tens_val(k,j,i) = M_tens_val(i,j,k);
                         end

                         for w=1:K
                             tidx = Tidx{k,w};
                             Tmp(:,:,w) = full(Tmp_mats{i}(:,tidx)*Tmp_mats_transp{j}(tidx,:));
                         end
                         %Tmp = ktensor({full(Tmp_mats{i}),full(Tmp_mats{j}),full(Tmp_mats{k})});
                         Tmp = Tmp./N_valid;
                         M_tens{i,j,k} = Tmp;
                         M_tens{i,k,j} = permute(Tmp,[1 3 2]); 
                         M_tens{j,i,k} = permute(Tmp,[2 1 3]);
                         M_tens{j,k,i} = permute(Tmp,[2 3 1]);
                         M_tens{k,i,j} = permute(Tmp,[3 1 2]);
                         M_tens{k,j,i} = permute(Tmp,[3 2 1]);
                  end
           end
        end
    end
%     for i=1:length(M_mat)
%         M_temp=M_mat{i};
%         M_temp(M_temp==0)=1e-6;
%         M_mat_sum=sum(M_temp,"all");
%         M_temp = M_temp./M_mat_sum;
%         M_mat{i}=M_temp;
%     end
end




