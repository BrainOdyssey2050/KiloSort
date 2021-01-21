function [uNew, nSnew]= reduce_clusters0(uS, crit)
% This funciton is used to reduce similar projection-vectors in uS by measuring normalized distance. 
% Only the representative-projection-vectors, i.e., the index are in newind, will be kept and extracted.
% Please see my note5 GitHub - cortex-lab Kilosort page5for more details.
% Input:
%     uS: a subset of the projection vectors. Each projection vector corresponds to an isolated peaks
%   crit: a normalized distance criteron. a representative-projection-vector will represent all the projection vectors in uS
%         whose distance is smaller than the 'crit = 0.65'
% Output:
%   uNew: a set of representative-projection-vectors selected in uS by a distance criteron
%  nSnew: the number of projection vectors that can be represented by each representative-projection-vector

%% Calculate the normalized distance (from 0-1) between any pairs of projection vectors in uS, and save in cdot
% the inner-product between any pairs of projections
cdot = uS * uS';              

% compute L2-norm Square of each projection vector
% S = sum(A,dim) returns the sum along dimension dim. For example, if A is a matrix, then sum(A,2) is a column vector containing the sum of each row.
newNorms  = sum(uS.^2, 2)';   

% compute sum of pairs of norms, see my note page5
% repmat(baseNorms', 1, numel(newNorms)): Create a 1-by-numel(newNorms) matrix whose elements contain the value baseNorms'
cNorms = 1e-10 + repmat(newNorms', 1, numel(newNorms)) + repmat(newNorms, numel(newNorms), 1);

% compute normalized distance between a pair of projection vectors
cdot = 1 - 2*cdot./cNorms;    % the diagnoal elements should be zero, since the distance with itself should be zero
cdot = cdot + diag(Inf * diag(cdot));    % why?????

%% Select the representative-projection-vectors based on the distance criterion, see my audio note 1
[cmin, newind] = min(single(cdot>crit),[],1);
% M = min(A,[],dim) returns the minimum element along dimension dim. 
%   For example, if A is a matrix, then min(A,[],2) is a column vector containing the minimum value of each row.

% (original comment)if someone else votes you in, your votee doesn't count
% (original comment)newind(ismember(1:nN, newind)) = [];

newind = unique(newind(cmin<.5));  
% C = unique(A) returns the same data as in A, but with no repetitions. C is in sorted order.

if ~isempty(newind)
    newind = cat(2, newind, find(cmin>.5));
else
    newind = find(cmin>.5);
end

uNew = uS(newind, :);  % save the Select the representative-projection-vectors
nNew = size(uNew,1);   % the num of Select the representative-projection-vectors

%% For each newly-selected projection vector in uNew, how many projection vectors in uS can it represents? -> nSnew
nSnew = merge_spikes0(uNew, zeros(nNew, 1), uS, crit); % note, the output of iNonMatch is ignored

