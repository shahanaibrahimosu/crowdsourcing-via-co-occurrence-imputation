function [M_mat,M_mat_unmask,M_mat_true,N_valid_mat] = mask_observed_marginals(M_mat,A,lambda,p,opts)
    m1=opts.m_star_1;
    m2=opts.m_star_2;
    method_no=opts.impute_method;
    M_mat_unmask = M_mat;
    M = length(M_mat);
    M_mat= cell(M,M);
    M_mat_true=cell(M,M);
    N_valid_mat = opts.N_valid_mat;
    K=length(lambda);
    if(method_no==0 || method_no==3)
        for i=1:M
            for j=i:M
                if(binornd(1,p)==1)
                    M_mat{i,j}= M_mat_unmask{i,j};
                    M_mat{j,i}= M_mat{i,j}';
                else
                    M_mat{i,j}=[];
                    M_mat{j,i}=[];
                    N_valid_mat(i,j)=0;
                    N_valid_mat(j,i)=0;
                end
            end
        end
    elseif(method_no==2)
        for i=1:M
            for j=i:M
                member_flag=ismember([i,j],[m1,m2]);
                if(sum(member_flag)==0)
                    if(binornd(1,p)==1)
                        M_mat{i,j}= M_mat_unmask{i,j};%  A{i}*diag(lambda)*A{j}';
                        M_mat{j,i}= M_mat{i,j}';
                    else
                        M_mat{i,j}=[];
                        M_mat{j,i}=[];
                        N_valid_mat(i,j)=0;
                        N_valid_mat(j,i)=0;
                    end
                else
                    M_mat{i,j}= M_mat_unmask{i,j};
                    M_mat{j,i}= M_mat{i,j}';
                end
            end
        end
    else
        warning("Incorrect impute method!!");
    end
    if(~isempty(A))
        for i=1:M
            for j=1:M
                M_mat_true{i,j}= A{i}*diag(lambda)*A{j}';
            end
        end
    end
    
end