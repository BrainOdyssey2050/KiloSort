function Wrot = whiteningLocal(CC, yc, xc, nRange)

Wrot = zeros(size(CC,1), size(CC,1));  % for example, Wrot.shape = 32*32, init with zeros

for j = 1:size(CC,1)                   % j from 1 to the total channel number
    % find nRange number of closest channels to channel j, and order their channel numbers in ascend order in term of distance 
    ds          = (xc - xc(j)).^2 + (yc - yc(j)).^2;  % for each current channel j,calculate the distance between channel j and all the channels.
    [~, ilocal] = sort(ds, 'ascend');                 % sort the distance in ascend order, and return the corresponding channel index.
    ilocal      = ilocal(1:nRange);                   % select the nRange number of electrodes (32) that are mostly close to electrode j.
    
    
    [E, D]      = svd(CC(ilocal, ilocal));  % Using CC to reconstruct a local CC, which is the covariance matirx of nRange number of closet channels in terms of channel j, then do SVD.
    D           = diag(D);                  % egenvalues
    eps         = 1e-6;
    wrot0       = E * diag(1./(D + eps).^.5) * E'; % ZCA
    
    Wrot(ilocal, j)  = wrot0(:,1);  % ?? there are 'size(CC)' number of rows in Wrot, 'ilocal' doesn't necessary equal to 'size(CC)'. 
                                    % Why only the first column is extracted in each wrot0 ?
                                    % in column j of Wrot, the un-assigned row, are still zeros as initial
end