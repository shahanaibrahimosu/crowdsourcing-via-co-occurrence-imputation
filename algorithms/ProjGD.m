function H = ProjGD(X,opts)
W = opts.W;
H = opts.H_init;
epsilon=1/max(eig((W')*(W)));
H_prev=H;
for i=1:opts.max_iter
    H = H_prev- epsilon* ((W')*(W)*H_prev-W'*X);
    H=max(H,eps);   
    if((norm(H_prev-H)/length(X))<opts.tol)
         break;
    end
    H_prev=H;

end
end