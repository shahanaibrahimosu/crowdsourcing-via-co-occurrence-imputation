function UU = ProjGD_varaint(XX,W,UU,opts)
% Solving min sum_i||X_i-WU_i||_{F} subject to 1^T Ui=1^T
epsilon=1/max(eig((W')*(W)));
M = length(XX);
for m=1:M
    U=UU{m};X=XX{m};U_prev=UU{m};
    for i=1:opts.max_iter
            U = U_prev- epsilon* ((W')*(W)*U_prev-W'*X);
        switch opts.constraint
            case 'simplex_col'
                U = ProjectOntoSimplex(U',1);
                U = U';
            case 'nonnegative'
                U = max(U,0);
                U = min(U,1);
        end
        %U=max(U,0);   
        
        if((norm(U_prev-U)/length(X))<opts.tol)
            break;
        end
        U_prev=U;
    end
    UU{m}=U';
end