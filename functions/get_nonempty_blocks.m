function [X_comb,X_comb_row_n,list_f] = get_nonempty_blocks(Y,I)
N=length(I);
K=length(Y{1,2});
X_comb = cell(N,1);
X_comb_row_n = cell(N,1);
list_f = cell(N,1);
for i=1:N
    X=[];
    X_rn=[];
    list=[];
    for j=[1:i-1 i+1:N]
        if(rank(Y{i,j})== K)
            Y_rn = diag(1./sum(Y{i,j},2))*Y{i,j};
            X = [X Y{i,j}];
            X_rn = [X_rn Y_rn];
            list =[list j];
        end
    end
    X_comb{i}=X;
    X_comb_row_n{i}=X_rn;
    list_f{i} = list;
end
end