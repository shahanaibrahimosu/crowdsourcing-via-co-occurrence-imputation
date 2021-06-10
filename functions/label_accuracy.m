function [error,rate] = label_accuracy(y_est,y)
    valid_index = find(y>0);
    N= length(valid_index);
    u = find(y_est(valid_index)~=y(valid_index));
    error = length(u);
    rate = (length(u)/N);
end
