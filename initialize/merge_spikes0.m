function [nS, iNonMatch] = merge_spikes0(uBase, nS, uS, crit)
% INPUT:
% uBase: the selected representive-projection-vectors after reduction from uS, shape = (selected nums, 96)
%    nS: the storage for the num of the projection vectors that can be represented by each representive-projection-vector
%          for example, [32, 4, 56, 67, ..., 23]
%          means the first selected representive-projection-vector can represent 32 projection vectors (peaks)
%    uS: the fullset of projection vectors for the representive-projection-vectors selection
%  crti: the criteron for the selection, basically measuring the normalized distance between a pair of projection vectors

% OUTPUT:
%         nS: the num of the projection vectors that can be represented by each representive-projection-vector
%  iNonMatch: the index of the projection vectors in uS that can NOT be represented by the selected vectors in uBase

% From reduce_cluster0.m -> merge_spikes0.m
%  Input: nSnew = merge_spikes0(uNew, zeros(nNew, 1), uS, crit);

% From optimizePeaks.m -> merge_spikes0.m
% Input: uBase(1:ncurr,:), nS(1:ncurr), uS, ops.crit
    % where, initially
    %   uBase = zeros(1e4, nProj);       % shape=(10000,96) 
    %   nS = zeros(1e4, 1);              % shape=(10000,1) 
    %   ncurr = 1;                       %  
    %   uS: 4000 spikes' projections.    % shape=(4000, 96)
    %   ops.crit = .65;                  % upper criterion for discarding spike repeates (0.65)

% check % My note: 5.GitHub - cortex-lab Kilosort for detail. Also, please understand reduce_cluster0.m first!
%%
if ~isempty(uBase)
    % normalized distance calculation between a representive-projection-vector in uBase and a projection-vector in uS
    cdot = uBase * uS';    
   
    baseNorms = sum(uBase.^2, 2)';
    newNorms  = sum(uS.^2, 2)';
    cNorms = 1e-10 + repmat(baseNorms', 1, numel(newNorms)) + repmat(newNorms, numel(baseNorms), 1);
    
    cdot = 1 - 2 * cdot./cNorms;
    
    % 
    [cdotmin, imin] = min(cdot, [], 1);
    % cdotmin returns the min value in each column (across rows comparision)
    % imin returns the index of the corresponding min values.
    
    iMatch = cdotmin < crit;
    nSnew = hist(imin(iMatch), 1:1:size(uBase,1));
    nS = nS + nSnew';
    
    iNonMatch = find(cdotmin>crit); % find the index of the projection vectors in uS that can NOT be represented by the selected vectors in uBase
else
   iNonMatch = 1:size(uS,2); 
   nS = [];
end