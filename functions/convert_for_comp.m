function [A] = convert_for_comp(f)
%CONVERT_FOR_COMP Summary of this function goes here
%   Detailed explanation goes here

N = size(f,2);
M = size(f,1);

A = zeros(N*M,3);
for i=1:N
   indx1 = (i-1)*M + 1;
   indx2 = indx1 + M - 1;
   A(indx1:indx2,1) = i*ones(M,1);
   A(indx1:indx2,2) = (1:M)';
   A(indx1:indx2,3) = f(:,i);
end

indxx = find(A(:,3) > 0);
A = A(indxx,:);

end

