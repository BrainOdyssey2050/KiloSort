function rez = fitTemplates(rez, DATA, uproj)
% Input:
% rez: a structure constructed in preprocessData.m, storing all the config params
% DATA: the filtered + whitened signals
% uproj: the projections of the segments of isolated peaks in DATA, on the three predefined temporal PCs
%                           each segment shape is 61*32 = time*channels, each segment corresponds to an isolated peak in the DATA
%                           each PC shape is 61*1
%         ---------- the structure of uproj ----------
%                          PC1           PC2           PC3
%                     CH1 ... CH32  CH1 ... CH32  CH1 ... CH32 = 96 Columns
%         peak1       
%          ...
%        peak70138
%                    row1-column1 means: for segment 1 (61*32), the content of PC1(1*32) for channel 1
%         --------------------------------------------
%        To some extent, you can understand the temporal PCs as significant contents, e.g., a frequency oscillation

%% Params settings
nt0             = rez.ops.nt0;       % nt0 = 61, the length of the segment
rez.ops.nt0min  = ceil(20 * nt0/61); % ? the minimum length of segment                                                                     ??????????

ops = rez.ops;                       % put rez.ops in the workspace

rng('default');                      % Control random number generator
rng(1);

Nbatch      = rez.temp.Nbatch;       % put in the workspace
Nbatch_buff = rez.temp.Nbatch_buff;  % put in the workspace

Nfilt 	= ops.Nfilt; %256+128;       % =64, number of clusters to use (2-4 times more than Nchan=32, should be a multiple of 32), check config_eMouse.m, STEP 5

ntbuff  = ops.ntbuff;                % =64, samples of symmetrical buffer(?) for whitening and spike detection, check config_eMouse.m      ??????????
NT  	= ops.NT;                    % put in the workspace

Nrank   = ops.Nrank;                 % =3, matrix rank of spike template model(?)                                                          ??????????
Th 		= ops.Th;                    % =[4,10,10], threshold for detecting spikes on template-filtered data(?)                             ??????????
maxFR 	= ops.maxFR;                 % =20000, maximum number of spikes to extract per batch

Nchan 	= ops.Nchan;                 % =32

batchstart = 0:NT:NT*(Nbatch-Nbatch_buff);

delta = NaN * ones(Nbatch, 1);       % = [NaN,...,NaN],shape=191*1, ? for what                                                             ??????????
iperm = randperm(Nbatch);            % randomly shuffle batches

%% Initial template calculation for each cluster, WUinit.shape=61*32*64=time*channels*clusters
switch ops.initialize
    % situation 1: the case that will be excuted in the eMouse example
    case 'fromData'                           
        WUinit = optimizePeaks(ops,uproj);    % optimizePeaks(ops,uproj)-> reduce_clusters0(uS, crit) -> merge_spikes0(uBase,nS,uS,crit)
        dWU    = WUinit(:,:,1:Nfilt);
        % (original comment) dWU = alignWU(dWU);
        
    % situation 2:  there could be pre-calculated initial scale kmeans results, so just load it.
    otherwise
        if ~isempty(getOr(ops, 'initFilePath', [])) && ~getOr(ops, 'saveInitTemps', 0)           
            load(ops.initFilePath);
            dWU = WUinit(:,:,1:Nfilt);
            
        % situation 3: not 'fromData', no uproj calculated; no initial results to load in; then calculate WUinit from scratch
        else                                  
            initialize_waves0;
            
            ipck = randperm(size(Winit,2), Nfilt);
            W = [];
            U = [];
            for i = 1:Nrank
                W = cat(3, W, Winit(:, ipck)/Nrank);
                U = cat(3, U, Uinit(:, ipck));
            end
            W = alignW(W, ops);
            
            dWU = zeros(nt0, Nchan, Nfilt, 'single');
            for k = 1:Nfilt
                wu = squeeze(W(:,k,:)) * squeeze(U(:,k,:))';
                newnorm = sum(wu(:).^2).^.5;
                W(:,k,:) = W(:,k,:)/newnorm;
                
                dWU(:,:,k) = 10 * wu;
            end
            WUinit = dWU;
        end
end

% Save the initial templates
if getOr(ops, 'saveInitTemps', 0)    % there is no field 'saveInitTemps' in ops, so return 0.-> don't save initial cluster centers (inital templates)
    if ~isempty(getOr(ops, 'initFilePath', [])) 
        save(ops.initFilePath, 'WUinit') 
    else
       warning('cannot save initialization templates because a savepath was not specified in ops.saveInitTemps'); 
    end
end
% in this block, we got dWU.shape = 61*32*64, the initial template 61*32 for each cluster.

%% Decomposition of the initial templates of the clusters

% 1. decompose dWU(:,:,i) for each cluster, to get temporal PCs -> W, and spatial PCs -> U, and other items
[W, U, mu, UtU, nu] = decompose_dWU(ops, dWU, Nrank, rez.ops.kcoords);
% OUTPUT:
% top-3 temporal PCs for 64 clusters, W.shape = 61*64*3
% top-3 spatial PCs for 64 clusters,  U.shape = 32*64*3
% mu.shape = 64*1, the L2 norm of the eigenvalues for each decomposition(for one cluster template)
% UtU.shape = 64*64, whether the inner product (between any pairs of the best spatial PC from two clusters) 
%                    is bigger than 1. If yes, then the corresponding logical value is 1
% nu.shape = 1*64,
%     for example, for the Top1 temporal principle component in cluster1 (as shown in my note)
%              1st element - 0; 
%              2nd - 0; 
%              3rd - 1st; 
%              4th - 2nd; 
%              ...; 
%              61th - 59th; 
%              0 - 60th; 
%              0 - 61th
%      then, squared and sum to get 1 value.
%      For the Top-2 and -3, we do the same, so we get two more values. 
%      finally, sum all the 3 values together -> nu(1) for cluster 1

% 2. get the freq contents for each temporal PC in each cluster
W0 = W;
W0(NT, 1) = 0;       % add 3-d zero matirx (NT-61)*64*3 behind W along time dimension, so, W0.shape=131136*64*3
                     % basically, add (NT-61) behind each temporal PC
fW = fft(W0, [], 1); % fft for each temporal PC with (NT-61) zeros, zeros are used to keep the frequency resolution.
                     % fft.shape = 131136*64*3 complex number, so 131136 point fft.
                     % fft stores the frequency components of each temporal PC
                     % BUT WHY?
                     % BECAUSE each PC is an unique temporal content! just like frequency conponents! 
                     % so, the fft of a temporal PC is a representation of the temporal PC
fW = conj(fW);       % Complex conjugate -> the actual freq content of Top1,2,3 temporal PC in 64 clusters

%% param setting before optimizing templates
nspikes = zeros(Nfilt, Nbatch);  % num of spikes for each cluster and each batch
lam =  ones(Nfilt, 1, 'single'); % large amp for each cluster, initial with zeros

freqUpdate = 100 * 4;            % in the main optimization loop, update the parameters every freqUpdate iterations
                                 % BUT! The maximum interval of param % updating is Nbatch (if freqUpdate> Nbatch)
iUpdate = 1:freqUpdate:Nbatch;   % start: interval: end, 1:400:191 =>1

dbins = zeros(100, Nfilt);       % shape=100*clusters
        % dbins:      cluster 1 ... 64
        %        bin
        %          1
        %        ...                    shape=100*64
        %        100
        %     I guess dbins means the distribution of the amplitudes of the spikes from a cluster 

dsum = 0;                        % ? 
miniorder = repmat(iperm, 1, ops.nfullpasses);  % iperm.shape = 1*191, randomly shuffle batches
                                                % miniorder becomes 1*(191*6)
i = 1; % first iteration

epu = ops.epu;

% ----- pmi -----
pmi = exp(-1./linspace(1/ops.momentum(1), 1/ops.momentum(2), Nbatch*ops.nannealpasses));  % ?
   % pmi = exp(-1./exp(linspace(log(ops.momentum(1)), log(ops.momentum(2)), Nbatch*ops.nannealpasses)));
   % pmi = exp(-linspace(ops.momentum(1), ops.momentum(2), Nbatch*ops.nannealpasses));
   % pmi  = linspace(ops.momentum(1), ops.momentum(2), Nbatch*ops.nannealpasses);
% ops.nannealpasses    = 4;            % should be less than nfullpasses (4)
% ops.momentum         = 1./[20 400];  % start with high momentum and anneal (1./[20 1000])
% linspace(20, 400, 764)

% ----- Thi -----
Thi  = linspace(ops.Th(1),ops.Th(2), Nbatch*ops.nannealpasses);  % ?
% threshold for detecting spikes on template-filtered data ([4 10 10])

% ----- lami -----
if ops.lam(1)==0
    lami = linspace(ops.lam(1), ops.lam(2), Nbatch*ops.nannealpasses);
else
    lami = exp(linspace(log(ops.lam(1)), log(ops.lam(2)), Nbatch*ops.nannealpasses));
end
% ops.lam: large means amplitudes are forced around the mean ([10 30 30])
% seems all the three sets of params use Nbatch*ops.nannealpasses as the number of elements
% pmi, Thi, lami are params of pm, Th, lam for ith iteration

if Nbatch_buff<Nbatch
    fid = fopen(ops.fproc, 'r');
end

%% Optimization Main Loop

nswitch = [0];
msg = [];
fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

while (i<=Nbatch * ops.nfullpasses+1)    % all the batches need to pass 'ops.nfullpasses+1' times
                                         % there are Nbatch * ops.nfullpasses+1 in total
    
    % 1. (original comment)======== set the annealing parameters ==========
    if i<Nbatch*ops.nannealpasses
        Th      = Thi(i);
        lam(:)  = lami(i);
        pm      = pmi(i);
    end
    
    Params = double([NT Nfilt Th maxFR 10 Nchan Nrank pm epu nt0]);  % some of the parameters change with iteration number
    % NT: Num of Time points
    % Nfilt: Num of clusters we want
    % Th: threshold for detecting spikes on 'template-filtered data'
    % maxFR: maximum number of spikes to extract per batch
    % 10:
    % Nchan: Num of Channels
    % Nrank: Num of principle components
    % pm: from pmi
    % epu: Inf
    % nt0: 61
    
    % 2. (original comment) ======= update the parameters every freqUpdate iterations =========
    if i>1 &&  ismember(rem(i,Nbatch), iUpdate) 
        % the division reminder of i/191 should be a member of iUpdate=1, in order to get into this if condition
        % so, i=191+1, 191*2+1, ..., 191*5+1, can go into this if branch. 
        
        dWU = gather_try(dWU);
        
        % (original comment) break bimodal clusters and remove low variance clusters
        if  ops.shuffle_clusters &&...                      % ops.shuffle_clusters = 1, allow merges and splits during optimization
                i>Nbatch && rem(rem(i,Nbatch), 4*400)==1    % (original comment) i < Nbatch*ops.nannealpasses
            [dWU, dbins, nswitch, nspikes, iswitch] = ...
                replace_clusters(dWU, dbins,  Nbatch, ops.mergeT, ops.splitT, WUinit, nspikes);
        end
        
        dWU = alignWU(dWU, ops);
        
        % (original comment) parameter update
        [W, U, mu, UtU, nu] = decompose_dWU(ops, dWU, Nrank, rez.ops.kcoords);
        
        if ops.GPU
            dWU = gpuArray(dWU);
        else
            W0 = W;
            W0(NT, 1) = 0;
            fW = fft(W0, [], 1);
            fW = conj(fW);
        end
        
        NSP = sum(nspikes,2);
        if ops.showfigures

            subplot(2,2,1)
            for j = 1:10:Nfilt
                if j+9>Nfilt;
                    j = Nfilt -9;
                end
                plot(log(1+NSP(j + [0:1:9])), mu(j+ [0:1:9]), 'o');
                xlabel('log of number of spikes')
                ylabel('amplitude of template')
                hold all
            end
            axis tight;
            title(sprintf('%d  ', nswitch));
            subplot(2,2,2)
            plot(W(:,:,1))
            title('timecourses of top PC')
            
            subplot(2,2,3)
            imagesc(U(:,:,1))
            title('spatial mask of top PC')
            
            drawnow
        end
        % (original comment) break if last iteration reached
        if i>Nbatch * ops.nfullpasses; break; end
        
        % (original comment) record the error function for this iteration
        rez.errall(ceil(i/freqUpdate))          = nanmean(delta);
        
    end
    
    % 3. ========== Load data for each batch ==========
    % step1. (original comment) ----- select batch and load from RAM or disk -----
    ibatch = miniorder(i);  % miniorder (1*1146) is the six copies iperm (1*191). iperm is randomly shuffled batch index. 
    if ibatch>Nbatch_buff
        offset = 2 * ops.Nchan*batchstart(ibatch-Nbatch_buff);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
        dat = DATA(:,:,ibatch);
    end
    
    % step2. (original comment) ----- move data to GPU and scale it -----
    if ops.GPU
        dataRAW = gpuArray(dat);
    else
        dataRAW = dat;
    end
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;   % why scale the data? 
    % now we have dataRAW.shape = 131136*32
    
    % 4. ===============================================    
    % (original comment) project data in low-dim space
    data = dataRAW * U(:,:);    % U: top-3 spatial PCs for 64 clusters,  U.shape = 32*64*3, 
                                % dataRAW.shape = 131136 * 32; 
                                % U(:,:).shape = 32*192, each column is a spatial PC for a cluster
                                %                     Top1       Top2        Top3
                                %           cluster 1 ... 64   1 ... 64   1 ... 64
                                %      channel 1
                                %         ...
                                %      channel 32
                                % therefore, data.shape = 131136*192
                                % each row represents the contents of the 192 spatial PCs for a time point
                                %                           Top1       Top2        Top3
                                %                 cluster 1 ... 64   1 ... 64   1 ... 64
                                %           time point 1
                                %         ...                     
                                %      time point 131136
                                
    if ops.GPU
        % (original comment) run GPU code to get spike times and coefficients
        [dWU, ~, id, x, Cost, nsp] =    mexMPregMU(Params, dataRAW,  W, data, UtU, mu, lam .* (20./mu).^2, dWU, nu);
        % OUTPUT: in the example, there are 799 spikes detected
        %   dWU: shape=61*32*64, the template 61*32 for each cluster?
        %    id: shape=799*1, the belonging of each spike?
        %     x: shape=799*1, the amplitude of each spike?
        %  Cost: shape=799*1, the cost of ?
        %   nsp: num of spikes for each cluster, shape=64*1
    else
        [dWU, ~, id, x, Cost, nsp] = mexMPregMUcpu(Params, dataRAW, fW, data, UtU, mu, lam .* (20./mu).^2, dWU, nu, ops);
    end
    
    dbins = .9975 * dbins;  % (original comment) this is a hard-coded forgetting factor, needs to become an option
    
    if ~isempty(id)
        % ----- (original comment) compute numbers of spikes => nspikes
        nsp                = gather_try(nsp(:));
        nspikes(:, ibatch) = nsp;
        %        ibatch 1 2 3 ... 191
        %  cluster 1
        %     ...
        %  cluster 64 
        % NOTE: ibatch is the rondomly selected batch number, so the column of nspike will not be updated sequentially
        
        % ----- (original comment) bin the amplitudes of the spikes => dbins
        xround = min(max(1, int32(x)), 100);
        % NOTE: C = max(A,B) returns an array with the largest elements taken from A or B.
        % so, max(1, int32(x)) should be at least 1. max(1, int32(x)).shape = 799*1
        % in general, this commend will limit the amplitude of each spike in 1-100
        
        dbins(xround + id * size(dbins,1)) = dbins(xround + id * size(dbins,1)) + 1;
        % dbins:      cluster 1 ... 64
        %        bin
        %          1
        %        ...                    shape=100*64
        %        100
        %     I guess dbins means the distribution of the amplitudes of the spikes from a cluster        
        % xround + id * size(dbins,1) = xround + id * 100, so id*100 helps jump across columns (clusters)
        
        % ----- (original comment) estimate cost function at this time step => delta
        delta(ibatch) = sum(Cost)/1e3;
    end
    
    % ----- (original comment) update status
    if ops.verbose  && rem(i,20)==1    % update every 20 batches
        nsort = sort(round(sum(nspikes,2)), 'descend');
        fprintf(repmat('\b', 1, numel(msg)));
        msg = sprintf('Time %2.2f, batch %d/%d,\r mean mu across 64 clusters: %2.2f,\r mean neg-err: %2.6f,\r Num of ToTal spikes detected (NTOT): %d,\r n100 %d,\r n200 %d,\r n300 %d,\r n400 %d\r', ...
                              toc, i,Nbatch*ops.nfullpasses,        nanmean(mu(:)),       nanmean(delta),   round(sum(nsort)), nsort(min(size(W,2), 100)), nsort(min(size(W,2), 200)), nsort(min(size(W,2), 300)), nsort(min(size(W,2), 400)));
        % mu.shape = 64*1, the L2 norm of the eigenvalues for each decomposition (for one cluster template)
        
        fprintf(msg);
    end
    
    % increase iteration counter
    i = i+1;
end

%%
% (original comment) close the data file if it has been used
if Nbatch_buff<Nbatch
    fclose(fid);
end

if ~ops.GPU
   rez.fW = fW; % save fourier space templates if on CPU
end
rez.dWU     = gather_try(dWU);
rez.nspikes = nspikes;
% %%