function [F,f] = simulate_missing_blocks_mod2(f,K,p_obs_blocks,n_groups,n_collaborator_list)

M = size(f,1);
N = size(f,2);
f_mod = zeros(size(f));
L = M*(M-1)/2; % Number of available blocks
marg = combnk(1:M,2); 
marg = num2cell(marg,2); 
Omega = L*p_obs_blocks; % No of observed blocks



selectd_blocks = randsample(L,Omega);% randomly select the observed blocks

N_ind_available = [1:N]; % avaialble samples to be labeled by pairs

% n_groups = 3;
% n_collaborator_list = [10,13,2];
avg_samples = floor(N/n_groups); % Avg number of samples assuming
                              % each block (pairs) label equal no
                              % samples
range_no_samples = [round(0.5*avg_samples):round(1.5*avg_samples)]; % creating a range 
                        % for different number of samples by pairs
                        
M_ind_available=[1:M];
for i=1:n_groups-1
    M_sel = randsample(M_ind_available,n_collaborator_list(i));
    N_sel = randsample(range_no_samples,1); % select a number of samples for 
                                        % this pair      
    N_sel_indices = randsample(N_ind_available,N_sel); % select the indices of the 
                                                    % samples  
    for j=M_sel
        f_mod(j,N_sel_indices) = f(j,N_sel_indices);
    end
    N_ind_available = setdiff (N_ind_available,N_sel_indices); % update the avaialble
                                                            % sample indices     
    M_ind_available = setdiff(M_ind_available,M_sel);
    
end
M_sel = M_ind_available;
N_sel_indices = N_ind_available;
for j=M_sel
    f_mod(j,N_sel_indices) = f(j,N_sel_indices);
end

if(length(M_ind_available)<n_collaborator_list(n_groups))
    len = n_collaborator_list(n_groups)-length(M_ind_available);
    M_sel = randsample(1:M,len);   
    for j=M_sel
        f_mod(j,N_sel_indices) = f(j,N_sel_indices);
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


