function [rez, DATA, uproj] = preprocessData(ops)
tic;
uproj = [];
ops.nt0 	= getOr(ops, {'nt0'}, 61);
% v = getOr(s, field, [default]) returns the 'field' of the structure 's' or 'default' 
%     if the structure is empty or the field does not exist.
% In this case, since the field doesn't exsist, getOr will return 61, to the new defined field ops.nt0


if strcmp(ops.datatype , 'openEphys')        % convert data, only for OpenEphys
   ops = convertOpenEphysToRawBInary(ops);   % ? Not familiar with the func convertOpenEphysToRawBInary(ops)
end

%% Channel Map loading, params configuration, and put params in the workspace to build structure 'rez'. 
%  So, from self-setting params 'ops' to structure 'rez'
if ~isempty(ops.chanMap)  % ops.chanMap = 'C:\SpikeSorting\KiloSort\eMouse\eMouse_simulation\chanMap.mat'
    if ischar(ops.chanMap)
        load(ops.chanMap);
        try
            chanMapConn = chanMap(connected>1e-6);  % connected in chanMap.mat: 34*1 logical. it is used to select channels. 
                                                    % connected>1e-6: if the numbers in the connected are positive values, 
                                                    %                 then the corresponding channels are selected.
                                                  
            xc = xcoords(connected>1e-6);           % xc, yc: x and y coordinates of each channel
            yc = ycoords(connected>1e-6);           %         are newly defined arrays, which do not exsist in the chanMap.mat
        catch
            chanMapConn = 1+chanNums(connected>1e-6);
            xc = zeros(numel(chanMapConn), 1);      % numel: return number of array elements
            yc = [1:1:numel(chanMapConn)]';
        end
        % try-catch-end executes the statements in the try block and catches resulting errors in the catch block. 
        % This approach allows you to override the default error behavior for a set of program statements
        % see https://www.mathworks.com/help/matlab/ref/try.html
        
        ops.Nchan    = getOr(ops, 'Nchan', sum(connected>1e-6));  % ops.Nchan = 32
        ops.NchanTOT = getOr(ops, 'NchanTOT', numel(connected));  % ops.NchanTOT = 34
        
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
else        % if ops.chanMap is empty, which means there is no way to find the channel map file, then we define the channel map and params by ourselves.
    chanMap  = 1:ops.Nchan;
    connected = true(numel(chanMap), 1);
    
    chanMapConn = 1:ops.Nchan;    
    xc = zeros(numel(chanMapConn), 1);
    yc = [1:1:numel(chanMapConn)]';
end

if exist('kcoords', 'var')
    kcoords = kcoords(connected);      % ? kcoords meaning. % Also selecting channels according to 'connected'
else
    kcoords = ones(ops.Nchan, 1);
end

NchanTOT = ops.NchanTOT;               % put the params in parameter structure 'ops' to workspace
NT       = ops.NT ;                    % ? NT Number of Time sample points

%% Building structure 'rez'
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

%% Hardware checking, memory, buffer, etc.
d = dir(ops.fbinary);                         % ? ops.fbinary, the raw dataset
ops.sampsToRead = floor(d.bytes/NchanTOT/2);

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

%% load data into patches, filter, compute covariance
% Step1. Filter construction
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2  % 1st, make sure 'fslow' is a field in the structure 'ops', fslow means low-pass freq 
                                             % 2nd, the half of the sample rate of the raw signals, ops.fs/2, should be bigger than fslow
                                             %    ,which is the constrain for fslow
                                             % However, for now, there is no 'fslow' in the ops
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
                                             % Note: the second param in the butter is normalized cutoff frequency Wn.
                                             % the freqs are normalized with fs/2
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high'); % This is the case for this demo.
                                                       % 3 order, high-pass butterworth with 200Hz cutoff freq
                                                       % b1 and a1 are transfer function coefficients of the FILTER
end

% Step 2. Load the raw data
fprintf('Time %3.0fs. Loading raw data... \n', toc);
fid = fopen(ops.fbinary, 'r');

% Step 3. Some configs before actual preporcessing
ibatch = 0;
Nchan = rez.ops.Nchan;

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

if ~exist('DATA', 'var')                               % DATA exsists with 131136*32*191 int16, for now. What's inside exactly? 
    DATA = zeros(NT, rez.ops.Nchan, Nbatch_buff, 'int16');
end

isproc = zeros(Nbatch, 1);

% Step 4
while 1
    ibatch = ibatch + ops.nSkipCov;
    
    offset = max(0, 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    if ibatch==1
        ioffset = 0;
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
    
    %         keyboard;
    
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
    
    switch ops.whitening
        case 'noSpikes'
            smin      = my_min(datr, ops.loc_range, [1 2]);
            sd = std(datr, [], 1);
            peaks     = single(datr<smin+1e-3 & bsxfun(@lt, datr, ops.spkTh * sd));
            blankout  = 1+my_min(-peaks, ops.long_range, [1 2]);
            smin      = datr .* blankout;
            CC        = CC + (smin' * smin)/NT;
            nPairs    = nPairs + (blankout'*blankout)/NT;
        otherwise
            CC        = CC + (datr' * datr)/NT;
    end
    
    if ibatch<=Nbatch_buff
        DATA(:,:,ibatch) = gather_try(int16( datr(ioffset + (1:NT),:)));
        isproc(ibatch) = 1;
    end
end
CC = CC / ceil((Nbatch-1)/ops.nSkipCov);
switch ops.whitening
    case 'noSpikes'
        nPairs = nPairs/ibatch;
end
fclose(fid);
fprintf('Time %3.0fs. Channel-whitening filters computed. \n', toc);
switch ops.whitening
    case 'diag'
        CC = diag(diag(CC));
    case 'noSpikes'
        CC = CC ./nPairs;
end

if ops.whiteningRange<Inf
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather_try(CC), yc, xc, ops.whiteningRange);
else
    %
    [E, D] 	= svd(CC);
    D = diag(D);
    eps 	= 1e-6;
    Wrot 	= E * diag(1./(D + eps).^.5) * E';
end
Wrot    = ops.scaleproc * Wrot;

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');
fidW    = fopen(ops.fproc, 'w');

if strcmp(ops.initialize, 'fromData')
    i0  = 0;
    ixt  = round(linspace(1, size(ops.wPCA,1), ops.nt0));
    wPCA = ops.wPCA(ixt, 1:3);
    
    rez.ops.wPCA = wPCA; % write wPCA back into the rez structure
    uproj = zeros(1e6,  size(wPCA,2) * Nchan, 'single');
end
%
for ibatch = 1:Nbatch
    if isproc(ibatch) %ibatch<=Nbatch_buff
        if ops.GPU
            datr = single(gpuArray(DATA(:,:,ibatch)));
        else
            datr = single(DATA(:,:,ibatch));
        end
    else
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
    
    datr    = datr * Wrot;
    
    if ops.GPU
        dataRAW = gpuArray(datr);
    else
        dataRAW = datr;
    end
    %         dataRAW = datr;
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    if strcmp(ops.initialize, 'fromData') %&& rem(ibatch, 10)==1
        % find isolated spikes
        [row, col, mu] = isolated_peaks(dataRAW, ops.loc_range, ops.long_range, ops.spkTh);
        
        % find their PC projections
        uS = get_PCproj(dataRAW, row, col, wPCA, ops.maskMaxChannels);
        
        uS = permute(uS, [2 1 3]);
        uS = reshape(uS,numel(row), Nchan * size(wPCA,2));
        
        if i0+numel(row)>size(uproj,1)
            uproj(1e6 + size(uproj,1), 1) = 0;
        end
        
        uproj(i0 + (1:numel(row)), :) = gather_try(uS);
        i0 = i0 + numel(row);
    end
    
    if ibatch<=Nbatch_buff
        DATA(:,:,ibatch) = gather_try(datr);
    else
        datcpu  = gather_try(int16(datr));
        fwrite(fidW, datcpu, 'int16');
    end
    
end

if strcmp(ops.initialize, 'fromData')
   uproj(i0+1:end, :) = []; 
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

