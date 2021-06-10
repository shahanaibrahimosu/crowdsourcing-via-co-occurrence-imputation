function [Data,I] = LoadDataset_credit()
fid = fopen('UCI_CREDIT.data');
A = textscan(fid,'%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',' ');

n_samples = length(A{1});
n_features = size(A,2);
Data = zeros(n_samples,n_features);

%%%%%%1st attribute%%%%%

Data(A{1}==0,1)=1;
Data(A{1}==1,1)=2;

%%%%%%2nd attribute%%%%%%%%---15
min1 = min(A{2});
max1 = max(A{2});
step1 = 3;
range = min1:step1:max1;
count=0;
for k= 1:length(range)-1
    u=(A{2}>=range(k) & A{2} < range(k+1));
    if(sum(u)~=0)
        count=count+1;
        Data(u,2)=count;
    end
end
Data(A{2}>=range(length(range)),2)=count+1;

%%%%%%3rd attribute%%%%%%%% --10
min1 = min(A{3});
max1 = max(A{3});
step1 = 3;
range = min1:step1:max1;
count=0;
for k= 1:length(range)-1
    u=(A{3}>=range(k) & A{3} < range(k+1));
    if(sum(u)~=0)
        count=count+1;
        Data(u,3)=count;
    end
end
Data(A{3}>=range(length(range)),3)=count+1;

%%%%%%4th attribute%%%%%
Data(:,4)=A{4};


%%%%%%5th attribute%%%%%
range = 1:1:14;
count=0;
for k= 1:length(range)
    u=A{5}==k;
    if(sum(u)~=0)
        count=count+1;
        Data(u,5)=count;
        
    end
end
%%%%%%6th attribute%%%%%
range = 1:1:9;
count=0;
for k= 1:length(range)
    u=A{6}==k;
    if(sum(u)~=0)
        count=count+1;
        Data(u,6)=count;
        
    end
end

%%%%%7th attribute%%%%%%%%
min1 = min(A{7});
max1 = max(A{7});
step1 = 3;
range = min1:step1:max1;
count=0;
for k= 1:length(range)-1
    u=(A{7}>=range(k) & A{7} < range(k+1));
    if(sum(u)~=0)
        count=count+1;
        Data(u,7)=count;
    end
end
Data(A{7}>=range(length(range)),7)=count+1;

%%%%%%8th attribute%%%%%
Data(A{8}==0,8)=1;
Data(A{8}==1,8)=2;
%%%%%%9th attribute%%%%%
Data(A{9}==0,9)=1;
Data(A{9}==1,9)=2;
%%%%%%11th attribute%%%%%
Data(A{11}==0,11)=1;
Data(A{11}==1,11)=2;

%%%%%11th attribute%%%%%%%%
min1 = min(A{10});
max1 = max(A{10});
step1 = 3;
range = min1:step1:max1;
count=0;
for k= 1:length(range)-1
    u=(A{10}>=range(k) & A{10} < range(k+1));
    if(sum(u)~=0)
        count=count+1;
        Data(u,10)=count;
    end
end
Data(A{10}>=range(length(range)),10)=count+1;

%%%%%%12th attribute%%%%%
range = 1:1:3;
count=0;
for k= 1:length(range)
    u=A{12}==k;
    if(sum(u)~=0)
        count=count+1;
        Data(u,12)=count;
        
    end
end

%%%%%%15th attribute%%%%%
Data(A{15}==0,15)=1;
Data(A{15}==1,15)=2;

%%%%%13th attribute%%%%%%%%
min1 = min(A{13});
max1 = max(A{13});
step1 = 20;
range = min1:step1:max1;
count=0;
for k= 1:length(range)-1
    u=(A{13}>=range(k) & A{13} < range(k+1));
    if(sum(u)~=0)
        count=count+1;
        Data(u,13)=count;
    end
end
Data(A{13}>=range(length(range)),13)=count+1;

%%%%%14th attribute%%%%%%%%
min1 = min(A{14});
max1 = max(A{14});
step1 = 5000;
range = min1:step1:max1;
count=0;
for k= 1:length(range)-1
    u=(A{14}>=range(k) & A{14} < range(k+1));
    if(sum(u)~=0)
        count=count+1;
        Data(u,14)=count;
    end
end
Data(A{14}>=range(length(range)),14)=count+1;

%Remove rows having undefinied values
 for i =1:n_features
     Data(Data(:,i)==0,:)=[];
 end
 
 Data = Data(:,[n_features 2:n_features-1 1]);
    
I = zeros(1,n_features);
for i=1:n_features
    I(i) = length(unique(Data(:,i)));
end


end

