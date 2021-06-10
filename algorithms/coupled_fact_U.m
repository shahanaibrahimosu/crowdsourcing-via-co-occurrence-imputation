function U = coupled_fact_U(M_mat,U,ops)
M = length(M_mat);
K = size(M_mat{1,2},1);
max_iter=ops.max_iter;
lambda=ops.lambda;
p=ops.p;
epsilon=ops.epsilon;
for i=1:M
    if(isempty(U{i}))
        U{i} = randn(K,K);
    end
end

if(p==2)
    for iter=1:max_iter
        for i=1:M
            X = [];
            H = [];
            for j=[1:i-1 i+1:M]
                if(sum(M_mat{i,j},"all")~=0)
                    X = [X M_mat{i,j}];
                    H = [H U{j}'];
                end
            end
            U{i} = (X*H')*pinv(H*H'+lambda*eye(K));         
        end
    end
else
    for iter=1:max_iter
        for i=1:M
            X = [];
            H = [];
            for j=[1:i-1 i+1:M]
                if(sum(M_mat{i,j},"all")~=0)
                    w_j = (p/2)*(norm(M_mat{i,j}-U{i}*U{j}')^2+epsilon)^((p-2)/2);
                    X = [X sqrt(w_j)*M_mat{i,j}];
                    H = [H sqrt(w_j)*U{j}'];
                end
            end
            U{i} = (X*H')*pinv(H*H'+lambda*eye(K));         
        end 
    end
end

end