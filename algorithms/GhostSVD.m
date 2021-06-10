%===================== Ghosh-SVD ================
function [error_GhostSVD] = GhostSVD(A,y,valid_index,n,k,m)
t = zeros(n,k-1);
for l = 1:k-1
    O = zeros(n,m);
    for i = 1:size(A,1)
        O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    
    [U S V] = svd(O);
    u = sign(U(:,1));
    if u'*sum(O,2) >= 0
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_GhostSVD = mean(y(valid_index) ~= (J(valid_index)))
end