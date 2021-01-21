function [dWU, st, id, x,Cost, nsp] = mexMPregMUcpu(Params,dataRAW,fW,data,UtU,mu, lam , dWU, nu, ops)
% OUTPUT: in the example, there are 799 spikes detected
%   dWU: shape=61*32*64, the template 61*32 for each cluster?
%    id: shape=799*1, the belonging of each spike?
%     x: shape=799*1, the amplitude of each spike?
%  Cost: shape=799*1, the cost of ?
%   nsp: num of spikes for each cluster, shape=64*1

nt0 = ops.nt0;         % = 61
NT      = Params(1);   % = 131136
nFilt   = Params(2);   % = 64
Th      = Params(3);   % = 4, BTW, ops.Th=[4,10,10]

pm      = Params(8);   % = 0.9512, origin from pmi

%% from dataRAW ->data -> fdata -> proj -> Ci, but why?
% It seems this part is for spike detection from dataRAW using the temporal and spatial PCs we already got

% ===== 1. why do data -> fdata -> proj ? =====
fdata   = fft(data, [], 1);
% data.shape = 131136*192
% each row represents the contents of the 192 spatial PCs for a time point
% data                      Top1       Top2        Top3
%                 cluster 1 ... 64   1 ... 64   1 ... 64
%           time point 1
%         ...                     
%      time point 131136
% So, why do fft?
% data has already been decomposed in spatial, now decomposed in temporal?

% fdata.shape = 131136*192
% fdata                      Top1       Top2        Top3
%                 cluster 1 ... 64   1 ... 64   1 ... 64
%           freq 1
%         ...                     
%      freq 131136

proj    = real(ifft(fdata .* fW(:,:), [], 1));
% fW.shape = 131136*64*3, the freq contents of top-1,2,3 the temporal PC from 64 clusters
% fW(:,:).shape = 131136*(64clusters * Top3)
% fdata .* fW(:,:) Element-wise multiplication
% see Note for details

if ops.Nrank > 1    % ops.Nrank = 3
	proj    = sum(reshape(proj, NT, nFilt, ops.Nrank),3);
    % proj.shape = 131136*192
    % reshape(proj, 131136, 64, 3)
    % proj.shape = 131136 * 64
end

% ===== 2. Why do this calculations? =====
Ci = bsxfun(@plus, proj, (mu.*lam)');    % (mu.*lam)'.shape = 1*64
Ci = bsxfun(@rdivide, Ci.^2,  1 + lam'); % (1 + lam').shape = 1*64
Ci = bsxfun(@minus, Ci, (lam .* mu.^2)');% (lam .* mu.^2)'.shape = 1*64
% all the 3 calculations targeted clusters, since the shapes are the same. 

%% Optimization Process
[mX, id] = max(Ci,[], 2);
% mX.shape = 131136*1
% id.shape = 131136*1

% 1. ---------- spike time detection ----------
maX = -my_min(-mX, 31, 1);   % ?
id  = int32(id);             

st      = find((maX < mX + 1e-3) & mX > Th*Th);  % I guess, st is the detected spike time
% NOTE: k = find(X) returns a vector containing the linear indices of each nonzero element in array X.
% st.shape = num of detected spikes * 1
% I guess, for example, st = [2, 5, 24, ..., 345, ..., 123122, ...],
% only contains the starting spike times

st(st>NT-nt0) = [];    % I guess, we don't do anything to spikes that occur late than 131136-61, because there is no 61 time sample left

% 2. ---------- cluster assignment for spike times ----------
id      = id(st);

x = [];
Cost = [];
nsp = [];

if ~isempty(id)
    % the actual spike time periods
    inds = bsxfun(@plus, st', [1:nt0]'); 
    
    % get the spike raw waveform out via dataRAW(inds, :), and reshape to (time periods * num of spikes * num of channels)
    dspk = reshape(dataRAW(inds, :), nt0, numel(st), ops.Nchan);  
    % reshape the spike raw waveforms (time periods * num of channels * num of spikes), e.g, 61*32*799
    dspk = permute(dspk, [1 3 2]);
    
    x       = zeros(size(id));   % amplitude?
    Cost    = zeros(size(id));   
    % NOTE: size(id): num of detected spikes
    % x and Cost are for params for each detected spike
    
    nsp     = zeros(nFilt,1);    % the number of spikes for each cluster?
    
    for j = 1:size(dspk,3)       % for each detected spike
        % 3. ---------- update the cluster temporal-spatial template ----------
        % for each detected spike j -> find the assigned cluster id(j)
        % -> find the cluster temporal-spatial template dWU(:,:,id(j))
        % -> find the spike j temporal-spatial waveform dspk(:,:,j)
        % -> update the cluster temporal-spatial template using a param 'pm'
        dWU(:,:,id(j))  = pm * dWU(:,:,id(j)) + (1-pm) * dspk(:,:,j); 
        
        % 4. ---------- find the spike j amplitude ----------
        x(j) = proj(st(j), id(j));
        % now we know, proj stores the amplitude for different time point * clusters
        
        % 5. ---------- find spike j cost? ----------
        Cost(j) = maX(st(j));
        % now we know Max is related to Cost
        
        % 6. ---------- update the number of spikes for cluster id(j) ----------
        nsp(id(j))= nsp(id(j)) + 1;
    end
    
    id = id - 1;  % ?
end