function [Pm,A] = getPermutedMatrix(A,list_g)
% K = size(A{1},2);
% marg=num2cell(perms(1:K),2);
% d_sum = zeros(1,size(marg,1));
% for j=list_g
%     for i=1:size(marg,1)
%         I_m= eye(K);
%         Pm = I_m(:,marg{i});
%         d_sum(i)= trace(A{j}*Pm);
%     end
%     [~,max_id] = max(d_sum);
%     Pm =  I_m(:,marg{max_id});
%     A{j}=A{j}*Pm;
% end
Pm = eye(10);
A_p = A;

for j=list_g
    K=size(A{j},2);
    take_flag =zeros(K,1);
    fill_flag = zeros(K,1);
    A_p{j}=zeros(size(A{j}));
    for i=1:K
       [~,ind] = max(A{j}(i,:));
       if(fill_flag(i)==0)
        fill_flag(i)=1;
        A_p{j}(:,i) = A{j}(:,ind);
        take_flag(ind)=1;
       end
    end
    for i=1:K
        if(fill_flag(i)==0)
            while(1)
             sel=randsample(K,1);
             if(take_flag(sel)==0)
                 break;
             end
            end
            A_p{j}(:,i) = A{j}(:,sel);
        end       
    end
end
A=A_p;
end