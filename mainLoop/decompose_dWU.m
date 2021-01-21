function  [W, U, mu, UtU, nu] = decompose_dWU(ops, dWU, Nrank, kcoords)
% dWU is the initial template 61*32*64 for 64 clusters. shape = time * channels * clusters
% this function is used to decompose dWU(:,:,i) for each cluster, to get
% temporal PCs -> W, and spatial PCs -> U, and L2 norm of each decomposition egenvalues ->mu
% then calculate two additional items, U -> UtU and W -> nu.
% for now, I don't why do we need UtU and nu.
%% 1. initial spaces
[nt0 Nchan Nfilt] = size(dWU);

W = zeros(nt0, Nrank, Nfilt, 'single');    % shape = 61 * 3 * 64, the storage of the temporal priciples for each cluster template
U = zeros(Nchan, Nrank, Nfilt, 'single');  % shape = 32 * 3 * 64, the storage of the spatial priciples for each cluster template
mu = zeros(Nfilt, 1, 'single');            % shape = 64 * 1, 
% dmax = zeros(Nfilt, 1);

%% 2. get SVD decomposition of each template, dWU(:,:,k), to get temporal PCs W(:,:,k), spatial PCs U(:,:,k), and L2 norm of each decomposition egenvalues
dWU(isnan(dWU)) = 0;
if ops.parfor  % = 0. whether to use parfor to accelerate some parts of the algorithm. NO
    parfor k = 1:Nfilt
        [W(:,:,k), U(:,:,k), mu(k)] = get_svds(dWU(:,:,k), Nrank);
    end
else
    for k = 1:Nfilt
        [W(:,:,k), U(:,:,k), mu(k)] = get_svds(dWU(:,:,k), Nrank);  % 
    end
end
U = permute(U, [1 3 2]);  % U.shape=32*64*3
W = permute(W, [1 3 2]);  % W.shape=61*64*3

U(isnan(U)) = 0;

if numel(unique(kcoords))>1  % in this example, 0, so, do not excute.
    U = zeroOutKcoords(U, kcoords, ops.criterionNoiseChannels);
end
% OUTPUT:
% temporal PCs, W.shape = 61*64*3
% spatial PCs,  U.shape = 32*64*3
% mu.shape= 64*1

%% 3.U -> UtU and W -> nu claculation

UtU = abs(U(:,:,1)' * U(:,:,1)) > .1;
% UtU tells us whether the abs of the inner product (of a pair of BEST TOP1 spatial PCs from two clusters) is bigger than 0.1
% UtU means U times U?

Wdiff = cat(1, W, zeros(2, Nfilt, Nrank)) - cat(1,  zeros(2, Nfilt, Nrank), W);
% W.shape = 61*64*3, zeros.shape = 2*64*3           zeros.shape = 2*64*3    W.shape = 61*64*3,
% NOTE: C = cat(dim,A,B) concatenates B to the end of A along dimension dim when A and B have compatible sizes 
%                       (the lengths of the dimensions match except for the operating dimension dim).
% for example, for the Top1 temporal principle component in cluster1 (as shown in my note)
%              1st element - 0; 
%              2nd - 0; 
%              3rd - 1st; 
%              4th - 2nd; 
%              ...; 
%              61th - 59th; 
%              0 - 60th; 
%              0 - 61th
% Wdiff.shape = 63*64*3

nu = sum(sum(Wdiff.^2,1),3);
nu = nu(:);

% mu = min(mu, 200);