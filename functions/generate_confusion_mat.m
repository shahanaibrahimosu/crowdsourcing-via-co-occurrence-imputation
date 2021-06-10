function [Gamma] = generate_confusion_mat(M,K,reliability)
%GENERATE_CONFUSION_MAT Function to generate synthetic confusion matrices
%for better than random annotators.
%   Input: M - Number of annotators, i.e. how many matrices to create
%          K - number of labels
%reliability - Level of classifier reliability. If ==0 then there is no
%control over classifier reliabilities. If ==1 then all classifiers are
%better than random. If >1 then the higher value of reliability the better
%the classifiers are.
%  Output: Gamma - Mx1 cell containing M KxK matrices.
%           Gamma(i,j) = Pr( f(X) = i | Y = j)
%           The largest elements of each column of a confusion matrix should
%           be on the diagonal. Also columns should sum up to 1.
%           
%       Panagiotis Traganitis. traga003@umn.edu
if nargin < 3
    reliability = 1;
end

Gamma = cell(M,1); %preallocate cell of confusion matrices
%numL = 1/L;
if reliability == 1
    for i = 1:M
        Gamma{i} = zeros(K,K);
        for j=1:K
            y = rand(K,1);
            [mval,idx] = max(y);
            y(idx) = y(j); 
            y(j) = mval;
            y = y./sum(y); %make sure the column sums up to 1
            Gamma{i}(:,j) = y;
        end
    end
elseif reliability == 0
    for i = 1:M
        Gamma{i} = zeros(K,K);
        for j=1:K
            y = rand(K,1);
            y = y./sum(y); %make sure the column sums up to 1
            Gamma{i}(:,j) = y;
        end
    end
elseif reliability > 1
    for i = 1:M
        Gamma{i} = zeros(K,K);
        for j=1:K
            y = rand(K,1);
            [mval,idx] = max(y);
            y(idx) = y(j); 
            y(j) = mval*reliability;
    %         y = zeros(L,1);
    %         y(j) = numL + numL*rand(1); %make the largest element be in the diagonal
    %         y(setdiff(1:L,j)) = numL*rand(L-1,1); %create the remaining elements at random
            y = y./sum(y); %make sure the column sums up to 1
            Gamma{i}(:,j) = y;
        end
    end
    
elseif reliability == -1
    for i=1:M
       Gamma{i} = zeros(K,K);
       for j=1:K
          y = rand(K,1);
          [~,idx] = max(y);
          while idx == j
              y = rand(K,1);
              [~,idx] = max(y);
          end
          y = y./sum(y);
          Gamma{i}(:,j) = y;
       end
    end
elseif reliability == -2
    for i=1:M
       tmp_vec = rand(K,1);
       tmp_vec = tmp_vec./sum(tmp_vec);
       Gamma{i} = repmat(tmp_vec,1,K);
    end
    
elseif reliability == -3 % K annotators are better than random and the rest are random
    U=randsample(M,K);
    for i = 1:K
        Gamma{U(i)} = zeros(K,K);
        for j=1:K
            if(j==i)
                y = rand(K,1);
                [mval,idx] = max(y);
                y(idx) = y(j); 
                y(j) = mval;
                y = y./sum(y); %make sure the column sums up to 1
            else
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
            end
            Gamma{U(i)}(:,j) = y;
        end
    end
    for i = 1:M
        if(sum(find(U==i))==0)
            Gamma{i} = zeros(K,K);
            for j=1:K
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
                Gamma{i}(:,j) = y;
            end
        end
    end
elseif reliability == -4 %Adding a spammer
    U=randsample(M,K+1);
    for i = 1:K
        Gamma{U(i)} = zeros(K,K);
        for j=1:K
            if(j==i)
%                 while(1)
%                     y = rand(K,1);
%                     [y_s,~]=sort(y);
%                     if(sum(y_s(1:end-1)) < y_s(end))
%                         break;
%                     end
%                 end
                y=rand(K,1);
                [mval,idx] = max(y);
                y(idx) = y(j); 
                y(j) = mval;
                y = y./sum(y); %make sure the column sums up to 1
            else
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
            end
            Gamma{U(i)}(:,j) = y;
        end
    end
    Gamma{U(K+1)}=zeros(K,K);
    for j=1:K
        y = rand(K,1);
        [mval,idx] = min(y);
        y(idx) = y(j); 
        y(j) = mval;
%         y = zeros(L,1);
%         y(j) = numL + numL*rand(1); %make the largest element be in the diagonal
%         y(setdiff(1:L,j)) = numL*rand(L-1,1); %create the remaining elements at random
        y = y./sum(y); %make sure the column sums up to 1
        Gamma{U(K+1)}(:,j) = y;
    end
    for i = 1:M
        if(sum(find(U==i))==0)
            Gamma{i} = zeros(K,K);
            for j=1:K
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
                Gamma{i}(:,j) = y;
            end
        end
    end
elseif reliability == -5 % 1 annotators are identity than random and the rest are random
    U=randsample(M,1);
    Gamma{U} = eye(K);

    for i = 1:M
        if(sum(find(U==i))==0)
            Gamma{i} = zeros(K,K);
            for j=1:K
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
                Gamma{i}(:,j) = y;
            end
        end
    end
elseif reliability == -6 % 1 annotators is diagonally dominant and the rest are random
    U=randsample(M,1);
    for ii=1:length(U)
        Gamma{U(ii)} = eye(K)+rand(K,1)/4;
        Gamma{U(ii)}=Gamma{U(ii)}*diag(1./sum(Gamma{U(ii)},1));
    end
%     for j=1:K
%         
%         
%         while(1)
%             y = rand(K,1);
%             [y_s,~]=sort(y);
%             if(sum(y_s(1:end-1)) < y_s(end))
%                 break;
%             end
%         end
%         [mval,idx] = max(y);
%         y(idx) = y(j); 
%         y(j) = mval;
%         y = y./sum(y); %make sure the column sums up to 1
%         Gamma{U}(:,j) = y;
%     end
    for i = 1:M
        if(sum(find(U==i))==0)
            Gamma{i} = zeros(K,K);
            for j=1:K
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
                Gamma{i}(:,j) = y;
            end
        end
    end
elseif reliability == -7 % 1 annotators is diagonally dominant and the rest are random
    U=randsample(M,1);
    for ii=1:length(U)
        Gamma{U(ii)} = rand(K,K);
        Gamma{U(ii)}=bsxfun(@rdivide,Gamma{U(ii)},sum(Gamma{U(ii)}));
        [vl,indx] = max(Gamma{U(ii)}); %move maximum value to diagonal
        vl2 = diag(Gamma{U(ii)}); 
        Gamma{U(ii)} = Gamma{U(ii)} - diag(vl2) + diag(vl);
        for j=1:K
            Gamma{U(ii)}(indx(j),j) = vl(j);
        end
        Gamma{U(ii)} = bsxfun(@rdivide,Gamma{U(ii)},sum(Gamma{U(ii)}));
    end

    for i = 1:M
        if(sum(find(U==i))==0)
            Gamma{i} = zeros(K,K);
            for j=1:K
                y = rand(K,1);
                y = y./sum(y); %make sure the column sums up to 1
                Gamma{i}(:,j) = y;
            end
        end
    end
 
end
end

