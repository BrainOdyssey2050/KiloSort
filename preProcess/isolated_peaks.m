function [row, col, mu] = isolated_peaks(S1, loc_range, long_range, Th)
% loc_range = [3  1];
% long_range = [30  6];
% for example, S1.shape = 131136*32
smin = my_min(S1, loc_range, [1 2]);  % same shape as S1
peaks = single(S1<smin+1e-3 & S1<Th); % same shape as S1, only contain 0 and 1, which indicates the isolated peaks in the signal

sum_peaks = my_sum(peaks, long_range, [1 2]);  % same shape as S1
peaks = peaks .* (sum_peaks<1.2).* S1; % peaks like MASK?

peaks([1:20 end-40:end], :) = 0;

% please refer to 'isolated_peaks.jpg' in the same folder as function isolated_peaks, to see how the raw signal changes from
% raw - smin - peaks - sum_peaks - peaks
% The general point is, after the processing, only the peak values are left, and all others are zeros
% then, we output the row and column of the peaks
[row, col, mu] = find(peaks);
