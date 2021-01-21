function WUinit = optimizePeaks(ops, uproj)
% Select the 'prototype' projection vectors in uS that are representative, which are the initial candidates for cluster centers.
% Initialize (updating) the cluster centers using the representative-projection-vectors.
% Reconstruct the initial temporal-spatial templates for clusters.
% RETURN: WUinit.shape=61*32*64=time*channels*clusters

%% Param settings
nt0  = ops.nt0;

nProj  = size(uproj,2);          % the number of projections for each segment (each isolated spike). 
nspike = size(uproj, 1);         % the number of isoated spike peaks detected in preprocessData.m across batches, e.g., the 1st batch has 342
%         ---------- the structure of uproj ----------
%                          PC1           PC2           PC3
%                     CH1 ... CH32  CH1 ... CH32  CH1 ... CH32 = 96 Columns
%         peak1       | <---------     projection vector 1 --------------> |
%          ...
%        peak70138    | <--------- projection vector 70138 --------------> |

%                    row1 - column1 means: for segment 1 (61*32), the content of PC1(1*32) for channel 1
%        also note, for a isolated peak segment, 61*32 (time*channels),
%          1.  the time samples are from -20 -> peak time -> 40, so 61 in total
%          2.  not all the 32 channels are used. only the peak channel with
%              the closet 4 channels on the right and left are used(9 in total, at most, sometime could be less) 
%         --------------------------------------------

nSpikesPerBatch = 4000;                   % every iteration will process projection vectors from 4000 isolated spikes, that are 4000 rows of uproj each iteration.   
nbatch = ceil(nspike / nSpikesPerBatch);  % therefore, 18 batches in this example

uBase = zeros(1e4, nProj);                % shape=(10000,96), the storage for the selected representative-projection-vectors.
nS = zeros(1e4, 1);                       % The number of spikes that the corresponding representative-projection-vector can represent.
                                          % for example, [32, 4, 56, 67, ..., 23]
                                          %   so, the first selected representative-projection-vectors from uproj can represent 32 projection vectors (peaks)
ncurr = 1;                                

%% Select the 'prototype' projection vectors in uS that are representative, which are the initial candidates for cluster centers.
for ibatch = 1:nbatch    % each time processes 4000 project vectors in uproj
    % ----- merge in with existing templates -----
    
    % 1. get the projection vectors for each ibatch.
    [b, e] = range_idx(ibatch, nSpikesPerBatch, nspike);  % the begining and ending index of the projection vectors in uproj used for each ibatch
      % ibatch 1, b=1,e=4000,
      %        2,   2,  8000,
      % ...
      %       18,  18, 70138.
      % using nSpikesPerBatch = 4000 as interval   
    uS = uproj(b:e, :);  % for every iteration, get 4000 projection vectors (4000 spikes or isolated peaks).
    
    % 2. posthoc explaination after understanding reduce_cluster0.m and merge_spikes0.m
    
    [nSnew, iNonMatch] = merge_spikes0(uBase(1:ncurr,:), nS(1:ncurr), uS, ops.crit);
    nS(1:ncurr) = nSnew;
    % For the first iteration, uBase(1:ncurr,:) = uBase(1:1,:) represents a zero vector with the shape (1*96), 
    %                          therefore, NO projection vectors in uS can be represented based on the distance
    %                          criterion ops.crit. 
    %                            The output nSnew will be 1*1 vector with zero, means uBase(1:1,:) can present nothing. 
    %                            The output iNonMatch will be all the index of the projection vectors in the uS,
    %                          because no vector has been 'matched' with uBase(1:ncurr,:) 
    % After the first excution, go to reduce_clusters0. we will be back in the 2nd iteration. Then you will understand
    % why merge_spikes0 is put in front of reduce_clusters0 -- despite the fact that merge_spikes0 is excuted also in the
    % last line of reduce_clusters0.
    
    % For the second iteration, uBase(1:ncurr,:) is the representative-projection-vectors detected before
    %                           nS(1:ncurr) is the number of projection vectors in the previous batches of uS that be represented by the corresponding representative-projection-vector.
    %                                       is the current representation ability of each representative-projection-vector
    %                           uS is a NEW batch of projection vectors.
    % this time, merge_spikes0 tell us how many projection vectors in uS can be represented by the PREVIOUS SELECTED representative-projection-vectors
    % then, update nS->nSnew. Importantly, merge_spikes0 selects the
    % vectors in this new uS batch that can NOT be represented, and return to iNonMatch
    
    [uNew, nSadd] = reduce_clusters0(uS(iNonMatch,:), ops.crit);
    % Reduce similar projection-vectors in uS(iNonMatch,:) by comparing the normalized distance ops.crit.
    % Only the representative-projection-vectors (uNew) will be returned.
    % nSadd is the number of projection vectors in uS that be represented by the corresponding representative-projection-vector.
    
    % For the first excution, uS(iNonMatch,:) is all the projection vectors in uS in this batch, 
    
    % For the second excution, find NEW representative-projection-vector in the uS(iNonMatch,:),
    % then update uBase and nS below
    
    % ----- add new representative-projection-vectors to uNew to uBase -----
    uBase(ncurr + [1:size(uNew,1)], :) = uNew;
    % ----- add the number of projection vectors in uS that be represented by the corresponding representative-projection-vector -----
    nS(ncurr + [1:size(uNew,1)]) = nSadd;

    ncurr = ncurr + size(uNew,1);

    if ncurr>1e4
        break;
    end
end
nS = nS(1:ncurr);
uBase = uBase(1:ncurr, :);
[~, itsort] = sort(nS, 'descend');
% itsort: the index of the selected representative-projection-vector in uBase. 
%         Ranking criterion: the MORE a selected vector can represent , 
%                            the HIGHER the its index will rank in the itsort.

% OUTPUT of this block:
% uBase: the 'prototype' projection vectors in uS that are representative, which are the initial candidates for clusters. shape=76*96
%    nS: how many vectors in the uS can be reprsented by each 'prototype' projection vectors. shape=76*1

%% initialize (updating) the cluster centers using the representative-projection-vectors
% 1. --------------- get the top-Nfilt(64) representative-projection-vectors, do basic configs ---------------
Nfilt = ops.Nfilt;                            % =64, number of clusters to use (2-4 times more than Nchan=32, should be a multiple of 32)
                                              % why? make sure at least one cluster is responsible for one channel?
                                              
lam = ops.lam(1) * ones(Nfilt, 1, 'single');  % ops.lam = [5 5 5], ? large means amplitudes are forced around the mean; ones(Nfilt, 1, 'single').shape = 64*1
                                              % lam means large amplitude?
                                              % Since now we only have nS and uBase, what is the amplitude?

ind_filt = itsort(rem([1:Nfilt]-1, numel(itsort)) + 1);  % get Nfilt num of index of selected projection vectors
% in this example, rem([1:Nfilt]-1, numel(itsort)) => rem([1:64]-1, 76) => 
% apparently, the division reminder will be [1:64]-1, after +1 the top 64 vecters' index (regarding uBase) are in ind_filt
% however, if the num of vectors numel(itsort) in uBase is small than the cluster you want, Nfilt,
% for example, Nfilt=100, then rem([1:100]-1, 76), then division reminder is [0,...,75,0,1,...23], then +1, [1,...,76,1,1,...24], 
% so some index in uBase will be used twice

if ops.GPU
    U = gpuArray(uBase(ind_filt, :))';        % load the top-Nfilt(64) representative-projection-vectors, and transpose -> shape 96*64
else
    U = uBase(ind_filt, :)';
end

U = U + .001 * randn(size(U));                % Add some randomness to U, so all the elements are non-zeros; % .001 * randn(size(U)): a matrix of random small numbers
mu = sum(U.^2,1)'.^.5;                        % L-2 norm of each representative-projection-vector. shape=64*1

U = normc(U);                                 % U.shape = 96*64
                                              % Normalize columns of matrix U to a length of 1, the L2 norm of each representative-projection-vector will be 1
                                              % basically normalize each Normalize representive-projection-vectors

% up to here, we
% - select the top-Nfilt(64) representative-projection-vectors =>U shape=96*64
% - calculate each vector's L2 norm => mu shape=64*1
% - normalized each representative-projection-vectors to L2-norm 1 =>U shape=96*64

% the top-Nfilt(64) representative-projection-vectors in U are the initial candidates for cluster centers
% in the coming 10 iterations, these 64 cluster centers will be updated again and again, by using all the uS in each iteration
% 2. --------------- Update cluster centers ---------------
for i = 1:10

    dWU = zeros(Nfilt, nProj, 'single');            % shape = 64*96, store the temp updated cluster centers
    if ops.GPU
        nToT = gpuArray.zeros(Nfilt, 1, 'single');  % shape = 64*1,
    else
        nToT = zeros(Nfilt, 1, 'single');
    end
    
    % 2.1 calcuate the sumation of all the projection vectors for each cluster
    for ibatch = 1:nbatch
        % in each iteration, process all the batches in uS
        % ----- STEP1 get a batch (clip) with the shape of 4000*96 -----
        [b, e] = range_idx(ibatch, nSpikesPerBatch, nspike);  % the index of projection vectors in uS used for each ibatch  
          % ibatch 1, b=1,e=4000, (b means begining; e means ending)
          %        2,   2,  8000,
          % ...
          %       18,  18, 70138.
          % using nSpikesPerBatch = 4000 as interval 
          
        arr_rows = e - b + 1;
        
        if ops.GPU
            clips = reshape(gpuArray(uproj(b:e, :)), arr_rows, nProj);  % get a batch of the projection vectors in uS, shape=4000*96
        else
            clips = reshape(uproj(b:e, :), arr_rows, nProj);
        end
        
        % ----- STEP 2 clustering each projection vector in clips to an initial cluster center, U -> cf -----
        ci = clips * U;  % clip.shape=4000*96, U.shape=96*64, -> 
                         % the inner product between all the projection vectors in clip and each representive-projection-vector in U
                         % ci.shape = 4000*64, each column belongs to one representive-projection-vector
        
        ci = bsxfun(@plus, ci, (mu .* lam)');
        % note: C = bsxfun(func,A,B) applies the element-wise binary operation specified by the function handle func to arrays A and B.
        % (mu .* lam)'.shape = 1*64, meaning?
        % add (mu .* lam)' to each row ci. BUT, why?
        
        cf = bsxfun(@rdivide, ci.^2, 1 + lam');  % (1 + lam').shape = 1*64
        cf = bsxfun(@minus, cf, (mu.^2.*lam)');  % (mu.^2.*lam)'.shape = 1*64
        % what's the meaning of the calculation? the meaning of ci and cf?
        % cf.shape=(4000,64), after calculation, the shape of cf is the same as ci
        %    each row is for one projection vector in uS, each column is related to one representative-projection-vector, 
        %    since top-64 are selected, the num of col is 64
        
        [~, id] = max(cf, [], 2);  % find max across rows, and return index to id, so each projection vector in uS get an id,
                                   % seems showing the 'belonging' of each projection vector
                                   % if my guessing is correct, then the above calculation in 2.2 
                                   % is a clustering process based on intial candidates of cluster centers in U
                                   % so, if the projection vector 5 should be in the cluster represented by U(:,10),
                                   % then, cf(5,10) must have the max across cf(5,:)
        id = gather_try(id);  % form GPU array to RAM array
        
        % see the note for the detail calculation
        
        % ------ STEP 3 assign projection vectors to a cluster, the results are in L with shape=(64, 4000) ------
        if ops.GPU
            L = gpuArray.zeros(Nfilt, arr_rows, 'single');  % define L with shape=(64, 4000)
        else
            L = zeros(Nfilt, arr_rows, 'single');
        end
        L(id.' + (0:Nfilt:(Nfilt * arr_rows - 1))) = 1;  
        % 0:Nfilt:(Nfilt * arr_rows - 1) = 0:64:(64*4000-1) = [0,64,128,192,...,64*4000-1],
        % which is a trick -> using one dim index to find elements in two dim matrix
        %   for example:
        %         test = zeros(5,2)
        %         test =
        %                0     0
        %                0     0
        %                0     0
        %                0     0
        %                0     0
        %         test(3) = 1
        %         test =
        %                0     0
        %                0     0
        %                1     0
        %                0     0
        %                0     0        
        % therefore, the interval 64 in 0:Nfilt:(Nfilt * arr_rows - 1) basically the index that helps jumping between columns of L(shape=64*4000)
        % id'.shape = 1*4000. id' selects one row in each column in L and assgin 1. 
        % It seems select one representive-projection-vectors among 64 for the corresponding projection vector in uS
        
        % ------ STEP 4 sum all the projection vectors for each cluster ---
        dWU = dWU + L * clips;  % L.shape=64*4000, clips.shape=4000*96
                                % each row of L * clips(64*96) is the sumation of all the projection vectors for each cluster
                                % dWU accumates each cluster's projection vectors across batches
                                % in this for loop, the cluster centers U are unchanged
        nToT = nToT + sum(L, 2);  % sum(L, 2).shape = 64*1
                                  % the number of projection vectors for each cluster
    end
    % OUTPUT of 2.1
    %  dWU:the sumation of all the projection vectors for each cluster, shape = 64*96
    % nToT:the number of projection vectors for each cluster, shape = 64*1
    
    % 2.2 get the new averaged cluster centers and update U
    dWU  = bsxfun(@rdivide, dWU, nToT); % dWU.shape = 64*96
    U = dWU';                           %   U.shape = 96*64
    
    mu = sum(U.^2,1)'.^.5;              % get the new L-2 norm of each cluster center. shape=64*1
    U = normc(U);                       % Normalize columns of matrix U to a length of 1, the L2 norm of each representative-projection-vector will be 1
end
% Keep updating the center of the clusters by averaging all the vectors in the clusters
% Updating 10 times in total

% In summary, this block uses the top-Nfilt(64) representative-projection-vectors as a start point of clustering, 
% Updating the center of the clusters 10 times, to get the initial cluster centers, U.shape = 96*64

%% reconstruct the temporal-spatial templates for clusters & extract temporal W and spatial U priciple components
Nchan = ops.Nchan;       % 32
Nfilt = ops.Nfilt;       % 64
Nrank = ops.Nrank;       % = 3, matrix rank of spike template model(?)
wPCA = ops.wPCA(:,1:3);  % shape=61*3
Urec = reshape(U, Nchan, size(wPCA,2), Nfilt);        % reshape(U, 32, 3, 64)
Urec= permute(Urec, [2 1 3]);                         % Urec.shape(3, 32, 64), PCs * Channels * Clusters
% We take Urec(:,:,i) as the representative-projection-vector for cluster i

% STEP1, reconstruct the normalized temporal-spatial templates from the updated initial cluster centers
Wrec = reshape(wPCA * Urec(:,:), nt0, Nchan, Nfilt);   % shape=61*32*64, time * channels * clusters
Wrec = gather_try(Wrec);

W = zeros(nt0, Nfilt, Nrank, 'single');               % shape=61*64*3, for the storage of temporal priciple components of clusters
U = zeros(Nchan, Nfilt, Nrank, 'single');             % shape=32*64*3, for the stroage of spatial priciple components of clusters

Wrec(isnan(Wrec(:))) = 0;  % if there is NaN in Wrec, change it to zero

% STEP2, the SVD decomposition of the temporal W and spatial U priciple components
for j = 1:Nfilt
    [w, sv, u] = svd(Wrec(:,:,j));  % Wrec(:,:,j) is the temporal-spatial template for cluster j
    % [U,S,V] = svd(A) performs a singular value decomposition of matrix A, such that A = U*S*V'.
    %    w     sv      u       Wrec(:,:,j)
    %   61*61  61*32  32*32       61*32
    
    w = w * sv;     % shape=61*32, for what?
    
    Sv = diag(sv);  % shape=32*1
    
    W(:,j,:) = w(:, 1:Nrank) / sum(Sv(1:ops.Nrank).^2).^.5;
    U(:,j,:) = u(:, 1:Nrank);
end
% STEP2 OUTPUT:
% W: a set of temporal priciple components for different clusters
% U: a set of spatial priciple components for different clusters
% However, these results have NOT been used as output or further process,
% so, why do step2?

% STEP3, get the actual scale of the temporal-spatial templates
mu = gather_try(single(mu));    % the L-2 norm of the averaged vector of each cluster center. shape=64*1
muinit = mu;
WUinit = zeros(nt0, Nchan, Nfilt);    % shape=61*32*64
for j = 1:Nfilt
    % multiply each (L-2 norm of the vector of a cluster) with (the temporal-spatial templates reconstructed by the vector)
    % Becuase the Urec(:,:,i) for template reconstruction is normalized,
    % the temporal-spatial templates reconstructed by this normalized vector, is also normalized.
    % therefore, the multiplication helps the temporal-spatial templates get back to the normal scale
    WUinit(:,:,j) = muinit(j)  * Wrec(:,:,j);
end
WUinit = single(WUinit);

end

function [b, e] = range_idx(idx, binsize, maxidx)
% idx = 1, binsize = 4000, maxidx = 70138
    b = min([(idx - 1) * binsize  + 1, maxidx]);
    e = min([idx * binsize, maxidx]);
end
