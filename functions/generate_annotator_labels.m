function [F,f] = generate_annotator_labels(Gamma,y,p_obs)
%GENERATE_ANNOTATOR_LABELS Function to generate annotator responses given
%ground truth labels and confusion matrices.
%  Input:Gamma - Mx1 cell containing LxL confusion matrices for each
%        annotator
%          y - Nx1 vector containing ground-truth labels.
%      p_obs - Mx1 vector containing annotator labeling probabilities
% Output:  F - Mx1 cell containing M LxN matrices, with annotator labels,
%          in vector format.
%          f - MxN matrix containing annotator labels in scalar format.
%          Each row corresponds to one annotator.
%   Panagiotis Traganitis. traga003@umn.edu

N = length(y);
M = size(Gamma,1);
K = numel(unique(y));

if nargin < 3
    p_obs = ones(M,1);
end

f = zeros(M,N); %annotator labels
for i=1:K
   idx = find(y == i); 
   for j=1:M
      n = numel(idx);
      tmp = randsample(K,n,true,Gamma{j}(:,i));
      f(j,idx) = tmp;
   end
end

for i=1:M
    mask = binornd(1,p_obs(i),1,N);
    f(i,:) = f(i,:).*mask;
end

F = cell(M,1); %cell of annotator responses. 

for i=1:M 
    indx = find(f(i,:) > 0);
    %N_i = numel(indx);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end

end

