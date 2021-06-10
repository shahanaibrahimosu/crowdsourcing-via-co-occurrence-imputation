function [vecA] = vec_func(A)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[I F] = size(A);
vecA=zeros(I*F,1);
for f=1:F,
    vecA(I*(f-1)+1:I*f)=A(:,f);
end

end

