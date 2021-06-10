function [U,time_coupled] = coupled_fact_U_constrained(M_mat,U,K,ops)
M = length(M_mat);
max_iter=ops.max_iter;
lambda=ops.lambda;
p=ops.p;
epsilon=ops.epsilon;
for i=1:M
    if(isempty(U{i}))
        U{i} = randn(K,K);
    end
end
Out.time_stamps = zeros(max_iter,1);
Out.hist_cost   = zeros(max_iter,1);
a=tic;
b = toc(a);
Out.time_stamps(1) = b;
Out.hist_cost(1) = cost(M_mat,U); 
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
            alpha = 1/max(eig((H)*(H')));
            U_prev = U{i};
            for it=1:max_iter
                U{i} = U{i}-alpha*(-X+U{i}*H)*H';
                U{i} = U{i}*diag(1./vecnorm(U{i}));
                if((norm(U_prev-U{i})/norm(U{i}))<ops.tol)
                    break;
                end
                U_prev = U{i};
            end 
        end
    end
else
    tic;
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
            alpha = 1/max(eig((H)*(H')));
            U{i} = (X*H')*pinv(H*H'+lambda*eye(K)); 
            U_prev = U{i};
            for it=1:5
                U{i} = U{i}-alpha*(-X+U{i}*H)*H';
                if (norm(U{i},'fro') > 1)
                    U{i} = (U{i}/norm(U{i},'fro'))*1;
                end
                if((norm(U_prev-U{i})/norm(U{i}))<ops.tol)
                    break;
                end
                U_prev = U{i};
            end            
        end 
        Out.time_stamps(iter) = toc(a);
        Out.hist_cost(iter) = cost(M_mat,U);
        if iter>1
            if (iter == max_iter ||  abs(Out.hist_cost(iter) - Out.hist_cost(iter-1))/abs(Out.hist_cost(iter-1)) < ops.tol)
                Out.iter = iter;
                Out.hist_cost(iter+1:end) = [];
                Out.time_stamps(iter+1:end)=[];
                break;
            end
        end
        a=tic;
    end
end
time_steps = cumsum(Out.time_stamps);
time_coupled=time_steps(end);
end
function c = cost(M_mat,U)
    c = 0;
    count=0;
    for i=1:length(M_mat)
        for j=1:length(M_mat)
            if(~isempty(M_mat{i,j}) || sum(M_mat{i,j},"all")~=0)
                c=c+norm(M_mat{i,j}-U{i}*U{j}','fro');
                count=count+1;
            end
        end
    end
    c=c./count;
end
