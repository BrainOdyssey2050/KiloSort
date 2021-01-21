function [W, U, mu] = get_svds(dWU, Nrank)
% INPUT:
%   dWU.shape = 61 * 32, an initial template for a cluster
%   Nrank = 3
% OUTPUT:
%   W: top-Nrank temporal PCs of the template dWU, shape=61*3
%   U: top-Nrank spatial PCs of the template dWU,  shape=32*3
%  mu: the L2-norm of top-Nrank engevalues         shape=1*1

% 1. Do an 'economic' SVD of an initial template for a cluster, see my note for details
[Wall, Sv, Uall] = svd(gather_try(dWU), 0);
% [U,S,V] = svd(A,0) produces a different economy-size decomposition of m-by-n matrix A:
%         m > n ¡ª svd(A,0) is equivalent to svd(A,'econ').
%                  [w,sv,u] = svd(dWU,'econ') produces an economy-size decomposition of m-by-n matrix A:
%                  m > n ¡ª Only the first n columns of w are computed, and sv is n-by-n.
%         m <= n ¡ª svd(A,0) is equivalent to svd(A).
%   dWU.shape = 61*32, so m > n. so 'econ' mode is used.
% OUTPUT:
% Wall.shape = 61*32, the Top-32 temporal priciple components
% Uall.shape = 32*32, all the spatial priciple components
% Sv.shape = 32*32, eign-values

% 2. find the max abs value in the first column of Wall,i.e, Wall(imax,1);
%    if Wall(imax,1) bigger than 1, then multiply Wall(imax,1) with the 1st
%    column of -Uall and -Wall, respectively. And take them as the new 1st
%    column of Uall and Wall. WHY ???
[~, imax] = max(abs(Wall(:,1)));  
Uall(:,1) = -Uall(:,1) * sign(Wall(imax,1));  % why Wall(imax,1) is so impartant? why multiplication?
Wall(:,1) = -Wall(:,1) * sign(Wall(imax,1));

    % ----- original comments -----
    %     [~, imin] = min(diff(Wall(:,1), 1));
    %     [~, imin] = min(Wall(:,1));
    %     dmax(k) = - (imin- 20);

    %     if dmax(k)>0
    %         dWU((dmax(k) + 1):nt0, :,k) = dWU(1:nt0-dmax(k),:, k);
    %         Wall((dmax(k) + 1):nt0, :)  = Wall(1:nt0-dmax(k),:);
    %     else
    %         dWU(1:nt0+dmax(k),:, k) = dWU((1-dmax(k)):nt0,:, k);
    %         Wall(1:nt0+dmax(k),:) = Wall((1-dmax(k)):nt0,:);
    %     end
    % ------------------------------

% 3. normalized the temporal principle components (32 in total) in Wall with their egenvalues
Wall = Wall * Sv;
Sv = diag(Sv);
mu = sum(Sv(1:Nrank).^2).^.5;
Wall = Wall/mu;

% 4. select top Nrank temporal and spatial components
% NOTE that since svd performs a singular value decomposition of matrix A, such that A = U*S*V'.
% V has a tranpose sign here, which means, each column of Uall (corresponding to V) is a spatial PC
W = Wall(:,1:Nrank);
U = Uall(:,1:Nrank);