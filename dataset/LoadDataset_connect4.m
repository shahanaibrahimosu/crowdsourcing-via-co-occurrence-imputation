function [Data,I] = LoadDataset_connect4()

fid = fopen('UCI_CONNECT4.data');
A = textscan(fid,'%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s','delimiter',',');
 %A = textscan(fid,'%s','delimiter',',');



n_samples = length(A{1});
n_features = size(A,2);
Data = zeros(n_samples,n_features);

for i=1:n_features-1
    range = ["x","o","b"];
    for k= 1:length(range)
        Data(strcmp(A{i},range(k)),i)=k;
    end
end


%%%%% Last attribute%%%%
range = ["win","loss","draw"];
Data(strcmp(A{n_features},range(1)),n_features)=1;
Data(strcmp(A{n_features},range(2)),n_features)=2;
Data(strcmp(A{n_features},range(3)),n_features)=3;


 %Remove rows having undefinied values
 for i =1:n_features
     Data(Data(:,i)==0,:)=[];
 end
 
Data = Data(:,[n_features 2:n_features-1 1]);
    
% I = zeros(1,n_features);
% for i=1:n_features
%     I(i) = length(unique(Data(:,i)));
% end
% 
% n_feat_step=2;
% n_feat_list = 1:n_feat_step:n_features-1;
% n_feat = length(n_feat_list);
% indicator = n_feat(end)~=n_features;
% if(indicator)
%     Data_mod = zeros(n_samples,n_feat+1);
% else
%     Data_mod = zeros(n_samples,n_feat);
% end
% 
% K=permn([1 2 3],n_feat_step);
% for i=1:n_feat-1
%     count=0;
%     for j=1:length(K)
%         u = sum(Data(:,n_feat_list(i):n_feat_list(i)+n_feat_step-1)==K(j,:),2)==n_feat_step;
%         if(sum(u)~=0)
%             count=count+1;
%             Data_mod(u,i)= count;
%         end
%     end
% end
% if(indicator)
%     Data_mod(:,n_feat+1)=Data(:,n_features);
%     K=permn([1 2 3],n_features-n_feat_list(end));
%     count=0;
%     for j=1:length(K)
%         u = sum(Data(:,n_feat_list(end):n_features-1)==K(j,:),2)==size(K,2);
%         if(sum(u)~=0)
%             count=count+1;
%             Data_mod(u,n_feat)= count;
%         end
%     end
% else
%    Data_mod(:,n_feat)=Data(:,n_features); 
% end
% 
% n_feat=size(Data_mod,2);
% 
% 
% Data_mod = Data_mod(:,[n_feat 2:n_feat-1 1]);

I = zeros(1,n_features);
for i=1:n_features
    I(i) = length(unique(Data(:,i)));
end

%Data=Data_mod;

end