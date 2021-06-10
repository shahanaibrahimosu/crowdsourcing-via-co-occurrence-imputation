function valid_pair_list= get_valid_stat_pair(M_mat,m)

M=size(M_mat,1);

valid_pair_list=[];
for j=1:M
    if(~isempty(M_mat{m,j}) || sum(M_mat{m,j},"all")~=0 )
        valid_pair_list=[valid_pair_list j];
    end    
end

end