function [F] = simulate_missing_blocks_mod(f,K,p_obs_blocks)

M = size(f,1);
N = size(f,2);
f_mod = zeros(size(f));
L = M*(M-1)/2; % Number of available blocks
marg = combnk(1:M,2); 
marg = num2cell(marg,2); 
Omega = L*p_obs_blocks; % No of observed blocks



selectd_blocks = randsample(L,Omega);% randomly select the observed blocks

N_ind_available = [1:N]; % avaialble samples to be labeled by pairs

M_collaborate_list=cell(M,1);
div_factor=0;
for i=1:M
    for j=1:length(selectd_blocks)
        sel_block = selectd_blocks(j);
        if(marg{sel_block}(1)==i)
            M_collaborate_list{i} = [M_collaborate_list{i} marg{sel_block}(2)];
        end
    end
    if(~isempty(M_collaborate_list{i} ))
        div_factor=div_factor+1;
    end    
end


avg_samples = floor(N/div_factor); % Avg number of samples assuming
                              % each block (pairs) label equal no
                              % samples
range_no_samples = [round(0.5*avg_samples):round(1.5*avg_samples)]; % creating a range 
                        % for different number of samples by pairs
                        
for i=1:div_factor-1
    N_sel = randsample(range_no_samples,1); % select a number of samples for 
                                        % this pair      
    N_sel_indices = randsample(N_ind_available,N_sel); % select the indices of the 
                                                    % samples  
    f_mod(i,N_sel_indices) = f(i,N_sel_indices); % for the selected samples allow 
                                            % the pairs to label
    for j=M_collaborate_list{i}
        f_mod(j,N_sel_indices) = f(j,N_sel_indices);
    end
    N_ind_available = setdiff (N_ind_available,N_sel_indices); % update the avaialble
                                                            % sample indices 
end

N_sel_indices = N_ind_available;% select the remianing samples
f_mod(div_factor,N_sel_indices) = f(j,N_sel_indices); % for the selected samples allow 
                                           % the pairs to label
for j=1:M_collaborate_list{div_factor}
    f_mod(j,N_sel_indices) = f(j,N_sel_indices);
end

for i=1:length(marg)
    if(~ismember(i,selectd_blocks))
        j = marg{i}(1);
        k = marg{i}(2);
        indices=find(f_mod(j,:)~=0 & f_mod(k,:)~=0);
        f_mod(j,indices)=0;
        f_mod(k,indices)=0;             
    end        
end

f= f_mod;

F = cell(M,1); %cell of annotator responses. 

for i=1:M 
    indx = find(f(i,:) > 0);
    %N_i = numel(indx);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end




end


