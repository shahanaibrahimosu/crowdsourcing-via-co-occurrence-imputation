function [idx,idx_vec] = label_estimator(Gamma,Z,estimate_type,p_vec)
%LABEL_ESTIMATOR Function to estimate final label (out of L possible) for N data, given N
%labels from M annotators and the confusion matrices of these M annotators
% This function essentially implements a MAP/ML detector.
%   Input: Gamma - LxLxM tensor containing confusion matrices. The i-th frontal slab Gamma(:,:,i) is the 
%               confusion matrix of the i-th annotator 
%           Z  - LxNxM tensor containing the labels from the M annotators.
%           Each LxN matrix Z(:,:,i) contains the labels (in vector format)
%           of annoatator i for all N data, i.e. if L = 3 and annotator i
%           has given datum j label 3 then Z(:,j,i) = [0;0;1];
%
%estimate_type - 'ML' or 'MAP'  
%       p_vec  - Lx1 vector of prior probabilities for each class, i.e.
%       p_vec(i) = Prob(Y = i); required if estimate_type == 'MAP';
%
%
% Output: idx - Nx1 vector containing final labels (in scalar format)
%     idx_vec - LxN matrix containing final labels (in vector format)
%     (sparse)
%   Panagiotis Traganitis. traga003@umn.edu

if nargin < 3
    estimate_type = 'ML';
end
if nargin < 4 && strcmpi('map',estimate_type)
    error('Vector or prior probabilities needs to be provided for MAP estimation');
end

size_gamma=size(Gamma);

if(length(size_gamma)~=3) % not in tensor format
    Gamma_cell=Gamma;
    K= size(Gamma_cell{1},1);
    M = length(Gamma_cell);
    Gamma = zeros(K,K,M);
    for i=1:M
        Gamma(:,:,i) = Gamma_cell{i};
    end 
end



size_Z = size(Z);
if(length(size_Z)~=3) % not in tensor format
    Z_cell=Z;
    K= size(Z_cell{1},1);
    N = size(Z_cell{1},2);
    M = length(Z_cell);
    Z = zeros(K,N,M);
    for i=1:M
        Z(:,:,i) = Z_cell{i};
    end 
end
    
  
M = size(Gamma,3); %number of annotators
N = size(Z,2); %number of data
K = size(Z,1); %number of labels


Gamma(find(Gamma == 0)) = eps;
logGamma = log(Gamma); %take the elementwise logarithm of the confusion matrices
logGamma(isinf(logGamma)) = 0;
logGamma(isnan(logGamma)) = 0;

U = zeros(N,K);
for i=1:M
    U = U + Z(:,:,i)'*logGamma(:,:,i);
end
if strcmpi('map',estimate_type)
   logpvec = log(p_vec);
   U = bsxfun(@plus,U,logpvec'); %add effect of prior probabilities on each column of U
elseif ~strcmpi('ml',estimate_type)
   warning('Unknown estimator type - returning ML estimate');
end

[~,idx] = max(U,[],2); %take index of maximum of each row. These are the labels we are looking for.
idx_vec = sparse(1:N,idx,1,N,K)'; %Final labels in vector format.
end

