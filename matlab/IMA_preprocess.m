function [lfpout] = IMA_preprocess(lfpdat, badE, wc, bw_order, trialInfo, plotinfo)
%%
% This function preprocesses LFP data before PCA
%
% Inputs:
%   lfpdat: [ntrial, nchannel, ntime] - Raw data
%   badE: [nBadChannel] - Index number of bad electrode/ecog channel(s)
%   wc: [float] Butterworth filter cutoff frequency [Hz]
%   bw_order: [int] Butterworth filter order
%   trialInfo: [struct] Meta information about the experiment
%   plotinfo: [struct] Meta info about plot parameters
%
% Outputs:
%   lfpout: [nchannel, ntime] - Preprocessed data
%%

% Check if ECOG or SC32 channel to look at is a bad channel
if ~isempty(badE(badE == plotinfo.idx))
    error('This is a bad channel, pick a different channel');
end

% FFT plot x axis parameters
freq_delta = 1/size(lfpdat, 3)*trialInfo.Fs_lfp;
freq_axis = 0:freq_delta:trialInfo.Fs_lfp-1;

% Trial average
if ischar(trialInfo.avgdTrials) && strcmp(trialInfo.avgdTrials, 'all')
    lfpavg = squeeze(mean(lfpdat, 1));
elseif isnumeric(trialInfo.avgdTrials) && ...
        max(trialInfo.avgdTrials) <= size(lfpdat, 1) && ...
        min(trialInfo.avgdTrials) >= 1  && ...
        length(trialInfo.avgdTrials) == 1
    lfpavg = squeeze(lfpdat(trialInfo.avgdTrials, :, :));
elseif isnumeric(trialInfo.avgdTrials) && ...
        max(trialInfo.avgdTrials) <= size(lfpdat, 1) && ...
        min(trialInfo.avgdTrials) >= 1  
    lfpavg = squeeze(mean(lfpdat(trialInfo.avgdTrials(1):...
        trialInfo.avgdTrials(2),:,:), 1));
else
    error('Invalid trials to investigate');
end

% 5 Hz Low pass 4th order butterworth filter
lfpout = zeros(size(lfpavg));  % Initialize array

if length(wc) == 2
    [blfp, alfp] = butter(bw_order, wc./(trialInfo.Fs_lfp./2)); % Create Filter
elseif length(wc) == 1
    [blfp, alfp] = butter(bw_order, wc./(trialInfo.Fs_lfp./2), 'low'); % Create Filter
else
    error('Invalid Butterworth filter cutoff frequencies');
end

% Apply LPF on data from good electrodes
for ii = setdiff(1:size(lfpdat, 2),badE)
    lfpout(ii,:) = filtfilt(blfp, alfp, lfpavg(ii,:));
end

% Include NaN values to mark bad electrodes
lfpout(badE,:) = NaN;

% Plot
if plotinfo.showplots
    figure; hold on;
    plot(lfpout(plotinfo.idx, :), 'LineWidth', 1); plot(lfpavg(plotinfo.idx,:), 'LineWidth', 1);
    legend([num2str(wc) ' Hz Filtered'], 'ECoG');
    xlim([0 600]);
    title(['Single Channel ' plotinfo.ID ' #' num2str(plotinfo.idx)], 'FontSize', 16);
    xlabel('Time', 'FontSize', 14); ylabel('[mV]', 'FontSize', 14);
    if plotinfo.saveplots
        saveas(gcf, [plotinfo.p2s '/singleChan_' plotinfo.ID '.png']);
    end
    
    figure; hold on;
    plot(freq_axis, abs(fft(lfpout(plotinfo.idx, :))), 'LineWidth', 1); plot(freq_axis, abs(fft(lfpavg(plotinfo.idx, :))), 'LineWidth', 1);
    legend([num2str(wc) ' Hz Filtered'], 'ECoG');
    xlim([0 100]);
    xlabel('Frequency [Hz]', 'FontSize', 14);
    title(['Single Channel ' plotinfo.ID ' FFT #' num2str(plotinfo.idx)], 'FontSize', 16);
    if plotinfo.saveplots
        saveas(gcf, [plotinfo.p2s '/singleChan_' plotinfo.ID '_fft.png']);
    end
end