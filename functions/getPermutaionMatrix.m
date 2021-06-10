function [Pm,A] = getPermutaionMatrix(A)
K = size(A{1},2);
marg=num2cell(perms(1:K),2);
d_sum = zeros(1,size(marg,1));
max_id = zeros(1,size(A,1));
for j=1:size(A,1)
    for i=1:size(marg,1)
        I_m= eye(K);
        Pm = I_m(:,marg{i});
        d_sum(i)= trace(A{j}*Pm);
    end
    [~,max_id(j)] = max(d_sum);
end
Pm =  I_m(:,marg{mode(max_id)});

for n=1:size(A,1)    
    A{n} = A{n}*Pm;
end
end