function [Aperm,Pm]= perm2match(A,B)
% permute columns of A to match columns of B
% size(B) must be equal to size(A)
% Nikos Sidiropoulos, UMN, July 15, 2012

% Greedy algorithm:
% [I,F]=size(A);
% D=zeros(F,F);
% for f=1:F,
%     for g=1:F,
%         D(f,g)=norm(A(:,f)-B(:,g));
%     end
% end
% Aperm=zeros(I,F);
% Pm=zeros(F,F);
% for k=1:F, 
%      [f,g]=find(D == min(min(D)));
%      Aperm(:,g)=A(:,f);
%      Pm(f,g)=1;
%      D(f,:)=1/eps;
%      D(:,g)=1/eps;
% end
% 
% Use Jonker-Volgenant algorithm for linear assignment problem instead:
[rowsol,cost,v,u,costMat] = lapjv(-(B'*A));
[I,F]=size(A);
IFxF = eye(F);
Pm = IFxF(rowsol,:)'; 
Aperm=A*Pm;