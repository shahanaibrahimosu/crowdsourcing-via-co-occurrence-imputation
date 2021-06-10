function P = krp(A,B)

[~, F] = size(A);
[~, F1] = size(B);

if (F1 ~= F)
 disp('krp.m: column dimensions do not match!!! - exiting matlab');
 exit;
end

P = bsxfun(@times,reshape(B,[],1,F),reshape(A,1,[],F));
P = reshape(P,[],F);

end