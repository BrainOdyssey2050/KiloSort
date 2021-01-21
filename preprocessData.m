function [rez, DATA, uproj] = preprocessData(ops)
tic;

uproj = [];  % uproj is used to store the projections of (the segments with isolated peaks, shape=61*32, in the filtered-whitened signals) onto (3 predefined temporal PCs, 61*3)
             % as defined below, uproj.shape=(1000000,96), each row represents the projections on 3 temporal PCs for an isolated peak segment, 1*32 for each projection.
             % uproj can store the projections for 1000000 peaks, at most.
             
ops.nt0 	= getOr(ops, {'nt0'}, 61);  % v = getOr(s, field, [default]) returns the 'field' of the structure 's' or 'default' if the structure is empty or the field does not exist.
                                        % In this case, since the field doesn't exsist, getOr will return 61, to the new defined field ops.nt0
                                        
                                        % ops.nt0: the number of elements in each PC in ops.wPCA that will be extracted
                                        % the only usage in the code:
                                        %     ixt  = round(linspace(1, size(ops.wPCA,1), ops.nt0));    
                                        %     wPCA = ops.wPCA(ixt, 1:3);
                                        % since both size(ops.wPCA,1) and ops.nt0 = 61 in this example, basically, all the elements in each temporal PC will be used.
                                        % However, it is possible that ops.nt0 < 61, which means not all the elements will be used.
                                        % Foundamentally, ops.nt0 is defined according to the length of the the segments with isolated peaks

if strcmp(ops.datatype , 'openEphys')        % convert data, only for OpenEphys
   ops = convertOpenEphysToRawBInary(ops);   % ? Not familiar with the func convertOpenEphysToRawBInary(ops)                                                               ??????????
end

%% Channel Map, channel number,fs setting, and using params in the workspace to build structure 'rez'. 
% 1. Settings
if ~isempty(ops.chanMap)                            % ops.chanMap = 'C:\SpikeSorting\KiloSort\eMouse\eMouse_simulation\chanMap.mat'
    if ischar(ops.chanMap)
        load(ops.chanMap);
        % ----- Channel map setting ----- 
        try
            chanMapConn = chanMap(connected>1e-6);  % 'connected' in chanMap.mat: 34*1 logical. it is used to select channels. 
                                                    %  connected>1e-6: if the numbers in the connected are positive values, 
                                                    %                  then the corresponding channels are selected.
                                                  
            xc = xcoords(connected>1e-6);           %  xc, yc: x and y coordinates of each channel
            yc = ycoords(connected>1e-6);           %          are newly defined arrays, which are based on xcoods ycoords in the chanMap.mat
        catch                                       % ?                                                                                                                   ??????????
            chanMapConn = 1+chanNums(connected>1e-6);
            xc = zeros(numel(chanMapConn), 1);      %  numel: return number of array elements
            yc = [1:1:numel(chanMapConn)]';
        end
        % try-catch-end executes the statements in the try block and catches resulting errors in the catch block. 
        % This approach allows you to override the default error behavior for a set of program statements
        % see https://www.mathworks.com/help/matlab/ref/try.html
        
        % ----- channel number setting ----- 
        ops.Nchan    = getOr(ops, 'Nchan', sum(connected>1e-6));  % ops.Nchan = 32
        ops.NchanTOT = getOr(ops, 'NchanTOT', numel(connected));  % ops.NchanTOT = 34
        % ----- sample rate setting ----- 
        if exist('fs', 'var')
            ops.fs       = getOr(ops, 'fs', fs);
        end
        
    else    % if ops.chanMap is NOT a character array
        chanMap = ops.chanMap;
        chanMapConn = ops.chanMap;
        xc = zeros(numel(chanMapConn), 1);
        yc = [1:1:numel(chanMapConn)]';
        connected = true(numel(chanMap), 1);      
        
        ops.Nchan    = numel(connected);
        ops.NchanTOT = numel(connected);
    end
else       % if ops.chanMap is empty, which means there is no way to find the channel map file, then we define the channel map and params by ourselves.
    chanMap  = 1:ops.Nchan;
    connected = true(numel(chanMap), 1);
    
    chanMapConn = 1:ops.Nchan;    
    xc = zeros(numel(chanMapConn), 1);
    yc = [1:1:numel(chanMapConn)]';
end

if exist('kcoords', 'var')
    kcoords = kcoords(connected);      % ? kcoords meaning. % Also selecting channels according to 'connected'                                                           ??????????
else
    kcoords = ones(ops.Nchan, 1);
end

NchanTOT = ops.NchanTOT;               % put the params in parameter structure 'ops' to workspace
NT       = ops.NT ;                    % ? NT Number of Time sample points                                                                                               ??????????

% 2. Building structure 'rez'
rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;
if exist('xcoords')
   rez.xcoords = xcoords;              % xc is the channels after selection, xcoords are before selection
   rez.ycoords = ycoords;
else
   rez.xcoords = xc;
   rez.ycoords = yc;
end
rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 

%% Hardware checking, memory, buffer, etc.                                                                                                             ??????????
d = dir(ops.fbinary);                         % ops.fbinary, the raw dataset
ops.sampsToRead = floor(d.bytes/NchanTOT/2);  % based on the number of bytes in the raw data, calculate how many samples needed to be read

if ispc
    dmem         = memory;
    memfree      = dmem.MemAvailableAllArrays/8;
    memallocated = min(ops.ForceMaxRAMforDat, dmem.MemAvailableAllArrays) - memfree;
    memallocated = max(0, memallocated);
else
    memallocated = ops.ForceMaxRAMforDat;
end
nint16s      = memallocated/2;

NTbuff      = NT + 4*ops.ntbuff;
Nbatch      = ceil(d.bytes/2/NchanTOT /(NT-ops.ntbuff));
Nbatch_buff = floor(4/5 * nint16s/rez.ops.Nchan /(NT-ops.ntbuff)); % factor of 4/5 for storing PCs of spikes
Nbatch_buff = min(Nbatch_buff, Nbatch);

%% The KEY process is below
%% Step1. Filter construction
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2            % 1st, make sure 'fslow' is a field in the structure 'ops', fslow means low-pass freq 
                                                       % 2nd, the half of the sample rate of the raw signals, ops.fs/2, should be bigger than fslow
                                                       %    ,which is the constrain for fslow
                                                       % However, for now, there is no 'fslow' in the ops
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
                                                       % Note: the second param in the butter is normalized cutoff frequency.
                                                       % the freqs are normalized with fs/2
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high'); % This is the case for this demo.
                                                       % 3 order, high-pass butterworth with 200Hz cutoff freq
                                                       % b1 and a1 are transfer function coefficients of the FILTER
end

%% Step 2. Load the raw data
fprintf('Time %3.0fs. Loading raw data... \n', toc);
fid = fopen(ops.fbinary, 'r');

%% Step 3. Some configs before actual preporcessing
ibatch = 0;
Nchan = rez.ops.Nchan;

% Initialize the Covariace Matrix with zeros
if ops.GPU
    CC = gpuArray.zeros( Nchan,  Nchan, 'single');     % 32*32 GPU array,single
else
    CC = zeros( Nchan,  Nchan, 'single');
end

if strcmp(ops.whitening, 'noSpikes')
    if ops.GPU
        nPairs = gpuArray.zeros( Nchan,  Nchan, 'single');
    else
        nPairs = zeros( Nchan,  Nchan, 'single');
    end
end

if ~exist('DATA', 'var')                               
    % DATA exsists with 131136*32*191 int16, time sample points*channels*batches. There is an automatic process that I don't know
    DATA = zeros(NT, rez.ops.Nchan, Nbatch_buff, 'int16');
end

isproc = zeros(Nbatch, 1);  % the indicator showing you which batches have been processed using the below while 1 loop.

%% Step 4, batch filtering,  co-variance matrix CC calculation
while 1
    % ---------- set up the batches that need to pass the filtering -> co-variance matrix calculation process ----------
    ibatch = ibatch + ops.nSkipCov;                    % initial: ibatch=0.   
                                                       % ops.nSkipCov=1->compute whitening matrix from every N-th batch (1)
                                                       % since the shape of DATA is 131392 * 32 * 167 (time sample points * channels * batches) So, there are 167 batches in total?
    % Keep in mind, In this example, ops.nSkipCov=1, which mean every batch will go through this while 1 process, which includes 
    % 1) filtering and 
    % 2) the iterative calculation of the co-variance matrix CC--> for whitening later
    % HOWEVER, ops.nSkipCov could be other values, which means NOT ALL the batches will go through this process. NOT ALL the batches will be
    % filtered in this loop! So what happen to those un-filtered batches? In the final for loop (for ibatch = 1:Nbatch), the first (else) is
    % used to target this situation
    
    % ---------- Buffer setting? ----------
    offset = max(0, 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    if ibatch==1
        ioffset = 0;
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
    
    % keyboard;
    
    if isempty(buff)  % If nothing left in the buffer, then the while loop STOP!
        break;
    end
    
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    
    % ---------- Load data for each batch ----------
    if ops.GPU
        dataRAW = gpuArray(buff);      % 34 channels * 131392 time samples GPU array int16
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';                % 131392 * 34 GPU array int16
    dataRAW = single(dataRAW);         % 131392 * 34 GPU array single
    dataRAW = dataRAW(:, chanMapConn); % 131392 * 32 GPU array single
    
    % ---------- Filtering ----------
    datr = filter(b1, a1, dataRAW);    % Filtering first, 
    datr = flipud(datr);               % then, Flip array up to down
    datr = filter(b1, a1, datr);       % And, do butterworth again
    datr = flipud(datr);               % Filp back
    % bi-directional filter?                                                                                                                                             ??????????
    
    % ---------- Co-variance matrix calculation per batch, then iteratively update CC ----------
    switch ops.whitening               % default 'full', for 'noSpikes' set options for spike detection
        case 'noSpikes'                % ? when                                                                                                                          ??????????
            smin      = my_min(datr, ops.loc_range, [1 2]);
            sd = std(datr, [], 1);
            peaks     = single(datr<smin+1e-3 & bsxfun(@lt, datr, ops.spkTh * sd));
            blankout  = 1+my_min(-peaks, ops.long_range, [1 2]);
            smin      = datr .* blankout;
            CC        = CC + (smin' * smin)/NT;
            nPairs    = nPairs + (blankout'*blankout)/NT;
        otherwise
            CC        = CC + (datr' * datr)/NT;  % datr.shape = 131392x32 gpuArray single, Iteratively calculating covariance matrix CC.
    end
    
    if ibatch<=Nbatch_buff            % Nbatch_buff: the number of batches in the buffer
        DATA(:,:,ibatch) = gather_try(int16( datr(ioffset + (1:NT),:)));  % for each finished batch, we put it in the corresponding dimension in the DATA
        isproc(ibatch) = 1;  % initially, isproc is an array with a shape of 191*1, all zeros. When we process one batch in this while 1 loop, we register 0->1 in the corresponding position.
    end
end

% sometimes, CC calculation does need all the batches, nSkipCov controls how many batches will be skipped. by default, ops.nSkipCov=1 which means
% all the batches are considered. However, if ops.nSkipCov=3, that mean, only 3,6,9,... should be used. since CC needs to be divided by the number
% of used batches for normalization purpose, they used the calculation below.
CC = CC / ceil((Nbatch-1)/ops.nSkipCov);

switch ops.whitening
    case 'noSpikes'
        nPairs = nPairs/ibatch;
end
fclose(fid);
fprintf('Time %3.0fs. Channel-whitening filters computed. \n', toc);

switch ops.whitening
    case 'diag'
        CC = diag(diag(CC));  % diag(CC) finds the diagonal elements of CC, then the outer diag() uses these elements to build a diagonal matirx (other elements are zeros)
    case 'noSpikes'
        CC = CC ./nPairs;
end

%% STEP 5: Whitening Matrix Calculation using CC
if ops.whiteningRange<Inf  % ops.whiteningRange = 32
    % MATLAB represents infinity by the special value inf .Infinity results from operations like division by zero and overflow, 
    % which lead to results too large to represent as conventional floating-point values.
    % ? WHEN does ops.whiteningRange is bigger than infinity? In what situation?                                                                                            ??????????
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather_try(CC), yc, xc, ops.whiteningRange);  % ! CHECK func whiteningLocal in the preProcess folder
else
    %
    [E, D] 	= svd(CC);
    D = diag(D);
    eps 	= 1e-6;
    Wrot 	= E * diag(1./(D + eps).^.5) * E';
end
Wrot    = ops.scaleproc * Wrot;                                         % ? ops.scaleproc = 200, what for?                                                                  ??????????

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid     = fopen(ops.fbinary, 'r');
fidW    = fopen(ops.fproc, 'w');

%% STEP 6: predefined temporal Pricipal Components loading
if strcmp(ops.initialize, 'fromData')
    i0  = 0;                                                 % ?
    ixt  = round(linspace(1, size(ops.wPCA,1), ops.nt0));    % ops.wPCA: predefined 7 principal components, ops.nt0 is defined in the begining of this script
    wPCA = ops.wPCA(ixt, 1:3);                               % extract the top-3 PCs
    
    rez.ops.wPCA = wPCA; % write wPCA back into the rez structure
    uproj = zeros(1e6,  size(wPCA,2) * Nchan, 'single');     % for example, 1000000*96, check the first several lines
end

%% STEP 7: Filtering, Whitening, and Projections
for ibatch = 1:Nbatch
    % ----------1. Make sure all the batches have been filtered, for unfiltered batches, excute else ----------
    if isproc(ibatch) %ibatch<=Nbatch_buff,              % if ibatch has been processed in the while 1 loop above, i.e., filtering -> CC calculation
        if ops.GPU
            datr = single(gpuArray(DATA(:,:,ibatch)));   % then, load the batch from DATA 
        else
            datr = single(DATA(:,:,ibatch));
        end
    else
        % in the above while 1 loop, if ops.nSkipCov !=1,then NOT all the batch has been processed, which means, this 'else' branch will be used
        % the processing here is only about filtering.
        offset = max(0, 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
        if ibatch==1
            ioffset = 0;
        else
            ioffset = ops.ntbuff;
        end
        fseek(fid, offset, 'bof');
        
        buff = fread(fid, [NchanTOT NTbuff], '*int16');
        if isempty(buff)
            break;
        end
        nsampcurr = size(buff,2);
        if nsampcurr<NTbuff
            buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
        end
        
        if ops.GPU
            dataRAW = gpuArray(buff);
        else
            dataRAW = buff;
        end
        dataRAW = dataRAW';
        dataRAW = single(dataRAW);
        dataRAW = dataRAW(:, chanMapConn);
        
        datr = filter(b1, a1, dataRAW);
        datr = flipud(datr);
        datr = filter(b1, a1, datr);
        datr = flipud(datr);
        
        datr = datr(ioffset + (1:NT),:);
    end
    
    % ---------- 2. Whitening ----------
    datr    = datr * Wrot;    % datr.shape = 131136*32 gpuArray single, Wrot.shape = 32*32
    
    if ops.GPU
        dataRAW = gpuArray(datr);
    else
        dataRAW = datr;
    end

    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;       % the reverse operation of Wrot = ops.scaleproc * Wrot; ? why?                                                                 ??????????
    
    % ---------- 3. Project the segments with isolated peaks in the whitened signals onto 3 predefined temporal PCs ----------
    if strcmp(ops.initialize, 'fromData') %&& rem(ibatch, 10)==1   %  ops.initialize could be 'fromData' or 'no'	
        % find isolated spikes, and return the row and column number of the peak points
        % check the function for details
        [row, col, mu] = isolated_peaks(dataRAW, ops.loc_range, ops.long_range, ops.spkTh);
        
        % find their PC projections
        % dataRaw.shape = (131136,32) = time points, channels
        % wPCA.shape    = (61,3)      = time,num of PCs
        % row.shape = col,shape = 342*1, this means, in the raw data, there are 342 isolated peaks in total
        uS = get_PCproj(dataRAW, row, col, wPCA, ops.maskMaxChannels);
        % uS.shape = (32,342,3)
        % see the figure get_PCproj in the same folder
        
        uS = permute(uS, [2 1 3]);
        % permute: B = permute(A,dimorder) rearranges the dimensions of an array in the order specified by the vector dimorder.
        % now, uS.shape = (342,32,3)
        
        uS = reshape(uS,numel(row), Nchan * size(wPCA,2));
        % in this example, numel(row) = 342, because there are 342 isolated peaks; size(wPCA,2) = 3, because there are top-3 temporal PCs
        % now, uS.shape = (342,96)
        
        % it seems uproj is used to store the projections above, according to the shape of uproj.shape=(1000000,96),
        % uproj can store at most 1000000 projection. 
        if i0 + numel(row) > size(uproj, 1)
            uproj(1e6 + size(uproj,1), 1) = 0;
        end
        uproj(i0 + (1:numel(row)), :) = gather_try(uS);
        % gather_try: a self-defined func that try to Transfer distributed array or gpuArray to local workspace
        
        i0 = i0 + numel(row); % i0 means the number of isolated peaks that have been processed in the past batches.
    end
    
    if ibatch<=Nbatch_buff
        DATA(:,:,ibatch) = gather_try(datr);
    else
        datcpu  = gather_try(int16(datr));
        fwrite(fidW, datcpu, 'int16');
    end
    
end

%% STEP 8 Additional operations after preprocessing
if strcmp(ops.initialize, 'fromData')
   uproj(i0+1:end, :) = [];     % for the unused part of the uproj (no projections assgined), =[]
end
Wrot        = gather_try(Wrot);
rez.Wrot    = Wrot;

fclose(fidW);
fclose(fid);
if ops.verbose
    fprintf('Time %3.2f. Whitened data written to disk... \n', toc);
    fprintf('Time %3.2f. Preprocessing complete!\n', toc);
end


rez.temp.Nbatch = Nbatch;
rez.temp.Nbatch_buff = Nbatch_buff;

