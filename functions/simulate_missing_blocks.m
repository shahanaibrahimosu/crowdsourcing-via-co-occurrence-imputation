function [F,f] = simulate_missing_blocks(f,K,p_obs_blocks)

M = size(f,1);
N = size(f,2);
f_mod = zeros(size(f));
L = M*(M-1)/2; % Number of available blocks
marg = combnk(1:M,2); 
marg = num2cell(marg,2); 
Omega = L*p_obs_blocks; % No of observed blocks

avg_samples = floor(N/Omega); % Avg number of samples assuming
                              % each block (pairs) label equal no
                              % samples
range_no_samples = [round(0.5*avg_samples):round(1.5*avg_samples)]; % creating a range 
                        % for different number of samples by pairs

selectd_blocks = randsample(L,Omega);% randomly select the observed blocks

N_ind_available = [1:N]; % avaialble samples to be labeled by pairs

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
    
    p_choice = 0.02;
    sel = binornd(1,p_choice);
    if(sel==1)
        N_sel_indices_add = randsample([1:N],N_sel);
        f_mod(j,N_sel_indices_add) = f(j,N_sel_indices_add); % for the selected samples allow 
                                                    % the pairs to label
        f_mod(k,N_sel_indices_add) = f(k,N_sel_indices_add);    
    end


    N_ind_available = setdiff (N_ind_available,N_sel_indices); % update the avaialble
                                                            % sample indices 
    if(length(N_ind_available) < round(1.5*avg_samples))
        N_ind_available=[N_ind_available 1:N];
    end
end

j = marg{selectd_blocks(end)}(1); % get the last indices of the pairs
k = marg{selectd_blocks(end)}(2);
N_sel_indices = N_ind_available;% select the remianing samples
f_mod(j,N_sel_indices) = f(j,N_sel_indices); % for the selected samples allow 
                                            % the pairs to label
f_mod(k,N_sel_indices) = f(k,N_sel_indices);

f= f_mod;

F = cell(M,1); %cell of annotator responses. 

for i=1:M 
    indx = find(f(i,:) > 0);
    %N_i = numel(indx);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end



end


