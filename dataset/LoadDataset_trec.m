function [F,f,y,K,M,N] = LoadDataset_trec()

fid = fopen('trec.txt');
A = textscan(fid,'%f%f%f','delimiter','\t');

fid = fopen('trec_truth.txt');
B = textscan(fid,'%f%f','delimiter','\t');

item_id = A{1};
annotator_id=A{2};
annotator_res=A{3};
ground_truth = B{2};

M = max(annotator_id);
N = max(item_id);
K = max(ground_truth);
f = zeros(M,N); %annotator labels

y = zeros(N,1);
y(B{1})=ground_truth;
% for i=1:N
%     ids = B{1}==i;
%     y(i,1) = mode(ground_truth(ids)); 
% end
C = [annotator_id item_id];

for i=1:length(C)
    f(C(i,1),C(i,2))=annotator_res(i);
end
% for i=1:N
%     g=f(:,i);
%     m=mode(g(g>0));
%     idx = g==0;
%     g(idx)=m;
%     f(:,i)=g;
%     conf_mat(m,i,idx)=0;
% end
% J=sum(f>0,1);
% [~,I]=sort(J,'descend');
% N=3000;
% f = f(:,I(1:N));
% 
% M=50;
% f=f(1:M,:);
% f_orig=f_orig(1:M,:);

% J=sum(f>0,2);
% [~,I]=sort(J,'descend');
% % 
% M=300;
% f=f(I(1:M),:);



F = cell(M,1); %cell of annotator responses. 

for i=1:M 
    indx = find(f(i,:) > 0);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end

end