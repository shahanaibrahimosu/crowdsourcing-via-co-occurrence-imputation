function [A_est_G,p_vec_hat,list_good_annot] = MultiSPA(M_mat,K,fix_permutation)

N = size(M_mat,1);
I= K*ones(1,N);

[X_cell,list_f] =get_nonempty_blocks(M_mat,I);
size_l=zeros(N,1);
for i=1:N
    size_l(i,1) = size(X_cell{i},2);
end
[~,index_spa] = max(size_l);
A_est_G = cell(N,1);


k=index_spa;
X=X_cell{k};


for k=1:N
    X=X_cell{k};
    if(size(X,2)>K)
        %%%%%%SPA Algorithm
        X = X*diag(1./sum(X,1));
        X(isnan(X))=0;
        [l,~,~]=FastSepNMF(X,K,0);
        W_est= X(:,l);
        A_est_G{k} = W_est;
        A_est_G{k} = A_est_G{k}*diag(1./sum(A_est_G{k},1));
    end
end


p_vec_hat = ones(K,1)/K; 

list_good_annot = [];
list_bad_annot = [];
for i=1:N
    if(~isempty(A_est_G{i}) && rank(A_est_G{i})==K)
        list_good_annot=[list_good_annot i];
    else
        list_bad_annot=[list_bad_annot i];
        A_est_G{i}=rand(K,K);
        A_est_G{i} = A_est_G{i}*diag(1./sum(A_est_G{i},1));
    end
end


 
if(fix_permutation)
    [~,A_est_G] = getPermutedMatrix(A_est_G,list_good_annot);
end

% count=0;
% p_vec_hat=zeros(K,1);
% for i=list_good_annot
%     for j=list_good_annot
%         if(i~= j && rank(M_mat{i,j})== K)
%             count=count+1;
%             A_e = kr(A_est_G{j},A_est_G{i});
%             temp=pinv(A_e)*M_mat{i,j}(:);
%             temp(temp<0)=0;
%             temp=temp./sum(temp);
%             temp(isnan(temp))=0;
%             p_vec_hat = p_vec_hat+temp;
%         end
%     end
% end
% p_vec_hat=p_vec_hat/count;
% p_vec_hat=p_vec_hat./sum(p_vec_hat);
% 
% 
% for i=1:length(A_est_G)
%     A_est_G{i};
%     G =A_est_G{i};
%     G = max( G, 10^-6 );
%     t=sum(G,1);
%     G = G*diag(1./t);
%     A_est_G{i}=G;
%     A_est_G{i};
% end

end