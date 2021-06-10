function [F] = simulate_missing_blocks_mod1(f,K,p_obs_blocks)

M = size(f,1);
N = size(f,2);
f_mod = zeros(size(f));
L = M*(M-1)/2; % Number of available blocks
marg = combnk(1:M,2); 
marg = num2cell(marg,2); 
Omega = L*p_obs_blocks; % No of observed blocks

avg_samples = floor(N/M); % Avg number of samples assuming
                              % each block (pairs) label equal no
                              % samples
range_no_samples = [0.5*avg_samples:1.5*avg_samples]; % creating a range 
                        % for different number of samples by pairs

selectd_blocks = randsample(L,Omega);% randomly select the observed blocks

N_ind_available = [1:N]; % avaialble samples to be labeled by pairs

N_flag = zeros(N,1);

for i=1:length(selectd_blocks)-1
    sel_block = selectd_blocks(i);
    j = marg{sel_block}(1); % get the indices of the pairs
    k = marg{sel_block}(2);
    N_sel = randsample(range_no_samples,1); % select a number of samples for 
                                            % this pair
    N_sel_indices = randsample(N_ind_available,N_sel); % select the indices of the 
                                                        % samples

    f_mod(j,N_sel_indices) = f(j,N_sel_indices); % for the selected samples allow 
                                                % the pairs to label
    f_mod(k,N_sel_indices) = f(k,N_sel_indices);
    N_flag(N_sel_indices)=ones(length(N_sel_indices),1);

%     N_ind_available = setdiff (N_ind_available,N_sel_indices); % update the avaialble
                                                            % sample indices                                                            
end

N_sel_indices_rem = N_sel_indices==0;% select the remianing samples

j = marg{selectd_blocks(end)}(1); % get the last indices of the pairs
k = marg{selectd_blocks(end)}(2);
%N_sel_indices = N_ind_available;% select the remianing samples
f_mod(j,N_sel_indices_rem) = f(j,N_sel_indices_rem); % for the selected samples allow 
                                            % the pairs to label
f_mod(k,N_sel_indices_rem) = f(k,N_sel_indices_rem);

f= f_mod;

F = cell(M,1); %cell of annotator responses. 

for i=1:M 
    indx = find(f(i,:) > 0);
    %N_i = numel(indx);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end



end


