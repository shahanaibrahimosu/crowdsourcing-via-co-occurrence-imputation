function [Data,I] = LoadDataset_car()
fid = fopen('UCI_CAR.data');
A = textscan(fid,'%s%s%s%s%s%s%s','delimiter',',');

dscr_features = 1:7;
c = length(dscr_features);

n_samples = size(A{1},1);
Data = zeros(n_samples,c);

Data( strcmp(A{1},'low')  ,1) = 1;
Data( strcmp(A{1},'med')  ,1) = 2;
Data( strcmp(A{1},'high') ,1) = 3;
Data( strcmp(A{1},'vhigh'),1) = 4;

Data( strcmp(A{2},'low')  ,2) = 1;
Data( strcmp(A{2},'med')  ,2) = 2;
Data( strcmp(A{2},'high') ,2) = 3;
Data( strcmp(A{2},'vhigh'),2) = 4;

Data( strcmp(A{3},'2')    ,3) = 1;
Data( strcmp(A{3},'3')    ,3) = 2;
Data( strcmp(A{3},'4')    ,3) = 3;
Data( strcmp(A{3},'5more'),3) = 4;

Data( strcmp(A{4},'2')    ,4) = 1;
Data( strcmp(A{4},'4')    ,4) = 2;
Data( strcmp(A{4},'more') ,4) = 3;

Data( strcmp(A{5},'small'),5) = 1;
Data( strcmp(A{5},'med')  ,5) = 2;
Data( strcmp(A{5},'big')  ,5) = 3;

Data( strcmp(A{6},'low') ,6) = 1;
Data( strcmp(A{6},'med') ,6) = 2;
Data( strcmp(A{6},'high'),6) = 3;

Data( strcmp(A{7},'unacc'),7) = 1;
Data( strcmp(A{7},'acc')  ,7) = 2;
Data( strcmp(A{7},'good') ,7) = 3;
Data( strcmp(A{7},'vgood'),7) = 4;

Data = Data(:,[c 1:c-1]);
I = zeros(1,c);
for i=1:c
    I(i) = length(unique(Data(:,i)));
end
end

