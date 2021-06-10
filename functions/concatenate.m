function [A_cat]= concatenate(A)
A_cat = [];
for n = 1:length(A)
    A_cat = [A_cat;A{n}];
end
end