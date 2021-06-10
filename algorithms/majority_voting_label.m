function [error_majority_vote] = majority_voting_label(Z,y,valid_index)
Z = Z(:,valid_index,:);
y = y(valid_index);
q = mean(Z,3);
%[I J] = max(q');
q=q';
accuracy = 0;
for j = 1:length(y)
    maxq = max(q(j,:));
    if y(j) > 0 && q(j,y(j)) == maxq
        accuracy = accuracy + 1 / size(find(q(j,:) == maxq), 2);
        %size(find(q(j,:) == maxq), 2)
    end
end
error_majority_vote = 1 - accuracy / size(y,1);
end