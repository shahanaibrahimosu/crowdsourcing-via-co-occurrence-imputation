function M_mat = alg_impute_blocks(M_mat,K,opts)
%% Get parameters
m1=opts.m_star_1;
m2=opts.m_star_2;
method_no=opts.impute_method;
p_block=opts.percent_blocks_to_impute;
M_mat_true=opts.M_mat_true;
M = size(M_mat,1);
N_mat_valid=opts.N_mat_valid;
%% Method 2
%Get the number of observed blocks
if(method_no==2)
    no_of_blocks=M*M-2;
    list_invalid=[];
    list_valid =[];
    count=0;
    for i=1:M
        for j=i:M
            cond = (i==m1 && j==m1) || (i==m2 && j==m2); %do not add super annotator diagonal matrix
            %since we will use a different scheme for imputing that
            if(~cond)
                if(sum(M_mat{i,j},"all")~=0)
                    if(i==j)
                        count=count+1;
                    else
                        count=count+2; % considering both M_mat{i,j} and M_mat{j,i}
                    end
                    member_flag=ismember([i,j],[m1,m2]);
                    if(sum(member_flag)==0)
                        list_valid =[list_valid;[i,j]]; % add the pair to observed index list
                    end
                else
                    list_invalid=[list_invalid;[i,j]]; % otherwise add to unobserved index list
                end
            end
        end
    end % count has the number of observed blocks
    
    
    len = length(list_invalid); % no of unobserved blocks
    ind_perm = randperm(len);
    list_invalid = list_invalid(ind_perm,:); % permute it 
    t=1;
    while(count < round(no_of_blocks*p_block)) % until the number of nonempty blocks meets the indented percentage
        ind=list_invalid(t,:); %first pair to be imputed
        M_mat{ind(1),ind(2)}=impute_block_mod(M_mat,ind(1),ind(2),m1,m2,K);
        M_mat{ind(2),ind(1)}=M_mat{ind(1),ind(2)}';
        if(~isempty(M_mat_true))
            if(norm(M_mat{ind(1),ind(2)}-M_mat_true{ind(1),ind(2)}) > 1e-1)
                warning("Incorrect imputation!!");
            end
            if(norm(M_mat{ind(2),ind(1)}-M_mat_true{ind(2),ind(1)}) > 1e-1)
                warning("Incorrect imputation!!");
            end
        end
        if(ind(1) == ind(2))
            count=count+1;
        else
            count=count+2;
        end
        t=t+1;
    end
    len = length(list_valid);
    ind_perm = randperm(len);
    M_mm1 = zeros(K,K);
    M_mm2 = zeros(K,K);
    n_block_impute=opts.no_blocks_imputation;
    count_imp=0;
    while(len >0 && count_imp < len)
        ind_l = list_valid(ind_perm(count_imp+1),:);
        M_mm1=M_mm1+impute_block_mod(M_mat,m1,m1,ind_l(1),ind_l(2),K);
        M_mm2=M_mm2+impute_block_mod(M_mat,m2,m2,ind_l(1),ind_l(2),K);
        count_imp=count_imp+1;
        if(count_imp==n_block_impute)
            break;
        end
    end
    M_mat{m1,m1} = M_mm1/count_imp;
    M_mat{m2,m2} = M_mm2/count_imp;
    if(~isempty(M_mat_true))
        if(norm(M_mat{m1,m1}-M_mat_true{m1,m1}) > 1e-1)
            warning("Incorrect imputation!!");
        end
        if(norm(M_mat{m2,m2}-M_mat_true{m2,m2}) > 1e-1)
            warning("Incorrect imputation!!");
        end
    end

%% Method 3 : impute blindly using all available satistics
elseif(method_no==3)
    n_block_impute=opts.no_blocks_imputation;
    no_of_blocks=M*M;
    count=0;
    list_invalid=[];
    list_valid =[];
    for i=1:M
        for j=i:M
            if(sum(M_mat{i,j},"all")~=0)
                if(i==j)
                    count=count+1;
                else
                    count=count+2; % considering both M_mat{i,j} and M_mat{j,i}
                end
                list_valid =[list_valid;[i,j]];
            else
                list_invalid=[list_invalid;[i,j]];
            end
        end
    end
    len = length(list_invalid);
    ind_perm = randperm(len);
    list_invalid = list_invalid(ind_perm,:);
    t=1;
    jjj=1;
    while(count < round(no_of_blocks*p_block))
        ind=list_invalid(t,:);
        m=ind(1); n=ind(2); % first pair to be imputed
        m_indices=find(N_mat_valid(m,:));
        n_indices=find(N_mat_valid(n,:));
        mn_valid_mat = N_mat_valid(m_indices,n_indices);
        [row,col]=find(mn_valid_mat);
        len_mn=length(row);
        ind_perm_row=randperm(len_mn);
        row=row(ind_perm_row);col=col(ind_perm_row);
        M_mn=zeros(K,K);
        count_imp=0;
        while(len_mn >0 && count_imp < len_mn)
            l1=m_indices(row(count_imp+1));
            l2=n_indices(col(count_imp+1));
            M_mn=M_mn+impute_block_mod(M_mat,m,n,l1,l2,K);
            count_imp=count_imp+1;
            if(count_imp==n_block_impute)
                break;
            end
            break;
        end
        M_mat{m,n}=M_mn;

        if(sum(M_mat{m,n},"all")~=0)
            jjj=jjj+1;
        end
        if(~isempty(M_mat_true))
            if(norm(M_mat{m,n}-M_mat_true{m,n}) > 1e-1)
                warning("Incorrect imputation!!");
            end
        end
        M_mat{n,m}=M_mat{m,n}';
        if(m==n)
            count=count+1;
        else
            count=count+2;
        end
        t=t+1;
    end
else
    warning("Invalid imputtaion method!!!");
end

    

end