function [Y] = get_second_order_stat(M_mat,marg)
N = size(M_mat,1);
for i=1:N
    for j=1:N
        if(i~=j)
            M_mat{i,j}(M_mat{i,j}==0)=1e-6;
            M_mat{i,j}=M_mat{i,j}./sum(M_mat{i,j},'all');
        end
    end
end
Y  = cell(size(marg,1),1);
for i=1:size(marg,1)
    Y{i} = M_mat{marg{i}(1),marg{i}(2)};
end
end