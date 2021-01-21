function Us = get_PCproj(S1, row, col, wPCA, maskMaxChans)
% S1.shape = 131136*32, raw data
% row.shape = col.shape = 342*1, the postion of isolated peaks in S1
% wPCA.shape = 61*3, three predefined temporal PCs
% maskMaxChans = 5, ?

%%
[nT, nChan] = size(S1);      % nT=131136; nChan=32
dt = -21 + [1:size(wPCA,1)]; % dt=[-20:40]
inds = repmat(row', numel(dt), 1) + repmat(dt', 1, numel(row));
% 1. repmat(row', numel(dt), 1)
%       row'.shape = 1*342; numel(dt)=61;
%       resulting shape = 61*342, each row is a copy of row'
% 2. repmat(dt', 1, numel(row))
%       dt'.shape = 61*1; numel(row).shape = 342
%       resulting shape = 61*342, each column is a copy of dt'
% 3. ind, each column is the 61 index for an isolated peak,
%    from peak_position - 20 -> peak_postion -> peak_postion +40

clips = reshape(S1(inds, :), numel(dt), numel(row), nChan);
% S1(inds, :): extract the segmentd with isolated peaks, then
% reshape S1(inds, :) to 61*342*32

%%
mask = repmat([1:nChan], [numel(row) 1]) - repmat(col, 1, nChan);
% 1. repmat([1:nChan], [numel(row) 1]).shape = 342*32
%      each row is a copy of channel numbers
% 2. repmat(col, 1, nChan)
%      each column is a copy of col, indicating the column position of each isolated spike
% 3. mask: 
%      repmat([1:nChan], [numel(row) 1]):
%         (1,2,3,4,...32)
%               ...
%         (1,2,3,4,...32) 342*32
%      repmat(col, 1, nChan):
%          2 2 2 2 ...2
%          3 3 3 3 ...3
%          4 4 4 4 ...4
%              ...
%         30 30 30 30 ...30 342*32
%      SUB
%         -1  0  1  2       ... 30
%         -2 -1  0  1  2    ... 29
%         -3 -2 -1  0  1  2 ... 28
%                    ...
%         -29  ...   -2 -1 0  1  2
%       for each isolated peak (each row), zero indicates the peak occurred channel

Mask(1,:,:) = abs(mask)<maskMaxChans;
% for each row, at most 9 channels are marked with 1.

%% 
clips = bsxfun(@times, clips , Mask);
% clips.shape = 61*342*32; Mask.shape = 1*342*32
% for each segment, 61*32, only select the channels : peak channel-4 ->peak channel -> peak channel+4
% 9 channels at most, could be less than 9 for some peaks, if the peaksoccur at channel 1-4 or 29-32

Us = wPCA' * reshape(clips, numel(dt), []);
% reshape(clips, numel(dt), []):
%   |<------------    32   ------------>|
%   |<--342-->|<--342-->| ....|<--342-->|
%  1
% ...
% 61
% Us:
%   |<------------    32   ------------>|
%   |<--342-->|<--342-->| ....|<--342-->|
%  1
%  2
%  3

Us = reshape(Us, size(wPCA,2), numel(row), nChan);
% Us.shape = 3*342*32

Us = permute(Us, [3 2 1]);
% us.shape = 32*342*3
