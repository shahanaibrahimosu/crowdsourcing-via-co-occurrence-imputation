function [F,f,y,K,M,N] = LoadDataset_RTE()

fid = fopen('rte.txt');
A = textscan(fid,'%f%f%f','delimiter','\t');

fid = fopen('rte_truth.txt');
B = textscan(fid,'%f%f','delimiter','\t');

item_id = A{1};
annotator_id=A{2};
annotator_res=A{3};
ground_truth = B{2};

M = max(annotator_id);
N = max(item_id);
K = max(ground_truth);
f = zeros(M,N); %annotator labels

y = size(N,1);
for i=1:N
    ids = B{1}==i;
    y(i,1) = mode(ground_truth(ids)); 
end
C = [annotator_id item_id];

for i=1:length(C)
    f(C(i,1),C(i,2))=annotator_res(i);
end

 F = cell(M,1); %cell of annotator responses. 
for i=1:M 
    indx = find(f(i,:) > 0);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end


end