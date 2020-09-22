%% ECoG Processing

% load some example recordings
global MONKEYDIR
MONKEYDIR = 'E:\aolab\data\centerOut_ECOG';

selectedfile = [MONKEYDIR, '\180328\posTargets_180328.mat'];
addpath(genpath([MONKEYDIR '/m/']))
addpath(genpath(['E:\aolab\codes\Si Jia\multiscale_analysis-master']))

% look at the sessions
drive_base = 'LM1_ECOG';
%only look at data from this day 180328
driveSessions = makeDriveDatabase(drive_base,{'180328','180328'});
Sessions = cat(1, driveSessions{:});
driveNames = Sessions(:,3);
driveNameAnalyze = {'LM1_ECOG_3'};
useSess          = ismember(driveNames, driveNameAnalyze);
DayAnalyze = Sessions(useSess,1);

% define processing information
trialInfo.sequence_random = 1;

%yes, we are filtering out the long acq time
trialInfo.filterAcq = 1;
trialInfo.filterAcq_time = 5000; %ms

trialInfo.trig = 'ReachStart'; %alignment signal
trialInfo.tBefore = -200; %ms
trialInfo.tAfter  = 400;%in mas
trialInfo.timeVec = linspace(trialInfo.tBefore,trialInfo.tAfter, trialInfo.tAfter - trialInfo.tBefore);
trialInfo.timeReachStart = abs(trialInfo.tBefore);

trialInfo.lfpType = 'lfp';
trialInfo.Fs_lfp = 1000; %sampling rate
trialInfo.badECoGs = [47 59 163]; %noting the noisy electrodes for later processing
trialInfo.ECoG_offset = 211;
trialInfo.goodECoGs = setdiff(1:211,trialInfo.badECoGs);


%we are going to look at all electrode,  211 ECOG and 32 SC32 electrodes
trialInfo.proc_electrodes = 1:243;

%use constant target defination
load(selectedfile)

%we load the regular trials
SessAnalyze = Sessions(useSess,:);
nSess = size(SessAnalyze,1);

for iD=1:nSess
    %load the trials and get the depth information
    trFN = [MONKEYDIR '/' DayAnalyze{iD} '/mat/Trials.mat'];
    load(trFN,'Trials')
    trialInfo.sessName =  DayAnalyze{iD};
    trialInfo.depthProfile = (Trials(1).Depth{1,2})'; % in micron
    trialInfo.goodSC32 = (find( trialInfo.depthProfile > 0))'; 
    trialInfo.goodE = [trialInfo.goodECoGs trialInfo.goodSC32+trialInfo.ECoG_offset];
    trialInfo.badE = setdiff(trialInfo.proc_electrodes,trialInfo.goodE);
    
    %filter out trials
    [trialInfo,Trials] = filter_trials(trialInfo,Trials);
    
    %first load an experiment trial
    expFN = [MONKEYDIR '/' DayAnalyze{iD} '/' Trials(1).Rec '/rec' Trials(1).Rec '.experiment.mat'];
    load(expFN,'experiment')
    
    %load data
    [trLfpData,trialInfo] = load_data(Trials, experiment,trialInfo,MONKEYDIR);
    
    trLfpData(:,trialInfo.badECoGs,:) = nan;
    
        %do target sorting
    [Pos, targids] = calcSeqReachTarg(Trials);
    Pos_Seq_1 = squeeze(Pos(:,2,:));
    trial_TargetAssigned = assignTaskNumber(Pos_Seq_1_unique,Pos_Seq_1);
end

%% Pre processing
p2s = 'E:/aolab/data/centerOut_ECOG/figures';   % Path to save

ECOG_chanidx = 100; % Define ECOG channel to look at
SC32_chanidx = 240; % Define SC32 channel to look at

wc = 50; % Cut off frequency [hz]
bw_order = 4;   % Butterworth filter order

% Check if ECOG or SC32 channel to look at is a bad channel
if ~isempty(trialInfo.badE(trialInfo.badE == ECOG_chanidx))
    error('Pick a different ECOG channel, this is a bad channel');
elseif ~isempty(trialInfo.badE(trialInfo.badE == SC32_chanidx))
    error('Pick a different SC32 channel, this is a bad channel');
end

% FFT plot x axis parameters
freq_delta = 1/size(trLfpData, 3)*trialInfo.Fs_lfp;
freq_axis = 0:freq_delta:trialInfo.Fs_lfp-1;

% Trial average
lfpavg = squeeze(mean(trLfpData, 1));

% Low pass 4th order butterworth filter
lfpfilt = zeros(size(lfpavg));  % Initialize array
[blfp, alfp] = butter(4,wc/(trialInfo.Fs_lfp/2), 'low'); % Create Filter
% Apply LPF on data from good electrodes
for ii = trialInfo.goodE
    lfpfilt(ii,:) = filtfilt(blfp, alfp, lfpavg(ii,:));
end

% Include NaN values to mark bad electrodes
lfpfilt(trialInfo.badE,:) = NaN;

% Plot
figure; hold on;
plot(lfpfilt(ECOG_chanidx, :), 'LineWidth', 1); plot(lfpavg(ECOG_chanidx,:), 'LineWidth', 1);
legend('5Hz Filtered', 'ECoG');
xlim([0 600]);
title('Single Channel ECoG', 'FontSize', 16); 
xlabel('Time', 'FontSize', 14); ylabel('[mV]', 'FontSize', 14);
saveas(gcf, [p2s '/singleChan_ECoG.png']);

figure; hold on; 
plot(freq_axis, abs(fft(lfpfilt(ECOG_chanidx, :))), 'LineWidth', 1); plot(freq_axis, abs(fft(lfpavg(ECOG_chanidx, :))), 'LineWidth', 1);
legend('5Hz Filtered', 'ECoG');
xlim([0 100]);
xlabel('Frequency [Hz]', 'FontSize', 14); 
title('Single Channel ECoG FFT', 'FontSize', 16);
saveas(gcf, [p2s '/singleChan_ECoG_fft.png']);

figure; hold on;
plot(lfpfilt(SC32_chanidx, :), 'LineWidth', 1); plot(lfpavg(SC32_chanidx,:), 'LineWidth', 1);
legend('5Hz Filtered', 'SC32');
xlim([0 600]);
title('Single Channel ECoG', 'FontSize', 16); 
xlabel('Time', 'FontSize', 14); ylabel('[mV]', 'FontSize', 14);
saveas(gcf, [p2s '/singleChan_SC32.png']);

figure; hold on; 
plot(freq_axis, abs(fft(lfpfilt(SC32_chanidx, :))), 'LineWidth', 1); plot(freq_axis, abs(fft(lfpavg(SC32_chanidx, :))), 'LineWidth', 1);
legend('5Hz Filtered', 'SC32');
xlim([0 100]);
xlabel('Frequency [Hz]', 'FontSize', 14); 
title('Single Channel ECoG FFT', 'FontSize', 16);
saveas(gcf, [p2s '/singleChan_SC32_fft.png']);
%% PCA

% Remove NaN values for processing
ECOG_pca.lfp = lfpfilt(trialInfo.goodECoGs,:);
SC32_pca.lfp = lfpfilt(trialInfo.goodSC32,:);

% Z-score
ECOG_pca.lfpz = zscore(ECOG_pca.lfp, 0, 'all');
SC32_pca.lfpz = zscore(SC32_pca.lfp, 0, 'all');

% Perform PCA
[ECOG_pca.coeff, ECOG_pca.score, ECOG_pca.latent, ~, ECOG_pca.explained] = pca(ECOG_pca.lfpz);

% Perform SVD
[ECOG_pca.u, ECOG_pca.s, ECOG_pca.v] = svd(ECOG_pca.lfpz', 'econ');
[SC32_pca.u, SC32_pca.s, SC32_pca.v] = svd(SC32_pca.lfpz', 'econ');

% Project ECOG data onto first principal components
ECOG_pca.p1 = diag(ECOG_pca.lfpz*repmat(ECOG_pca.u(:,1), [1, size(ECOG_pca.lfpz, 1)]));
ECOG_pca.p2 = diag(ECOG_pca.lfpz*repmat(ECOG_pca.u(:,2), [1, size(ECOG_pca.lfpz, 1)]));
ECOG_pca.p3 = diag(ECOG_pca.lfpz*repmat(ECOG_pca.u(:,3), [1, size(ECOG_pca.lfpz, 1)]));

% Calculate Variance
ECOG_pca.var = diag(ECOG_pca.s.^2);
SC32_pca.var = diag(SC32_pca.s.^2);

% Plot ECoG data projected onto the first 2 principal components
figure; hold on;
plot(ECOG_pca.p1, ECOG_pca.p2, 'k.', 'MarkerSize', 10);
% line([0 ECOG_pca.s(1)], [0 0], 'color', 'red', 'LineStyle', '--');
hold off;
xlabel('1', 'FontSize', 14); ylabel('2', 'FontSize', 14); 
% xlim([-30 30]); ylim([-15 15]);
title('ECoG - 2 PC', 'FontSize', 16)
saveas(gcf, [p2s '/ECoG_PCA2.png']);

% Plot ECoG data projected onto the first 3 principal components
figure; hold on;
plot3(ECOG_pca.p1,ECOG_pca.p2,ECOG_pca.p3, 'k.', 'MarkerSize', 10);
% line([0 ECOG_pca.s(1)], [0 0], [0,0], 'color', 'red', 'LineStyle', '--');
% line([0 0], [0 ECOG_pca.s(2)], [0,0], 'color', 'red', 'LineStyle', '--');
hold off;
xlabel('1', 'FontSize', 14); ylabel('2', 'FontSize', 14); zlabel('3', 'FontSize', 14); 
% xlim([-30 30]); ylim([-15 15]); zlim([-20 30]);
title('ECoG - 3 PC', 'FontSize', 16);
saveas(gcf, [p2s '/ECoG_PCA3.png']);

% Plot ECoG variance distribution across latent dimensions
figure; 
plot(100*cumsum(ECOG_pca.var./sum(ECOG_pca.var)), 'k', 'LineWidth', 1);
ylim([50 100]); ylabel('Cumulative Variance Explained [%]', 'FontSize', 14)
xlim([1 30]); xlabel('Latent Dimension Number', 'FontSize', 14);
saveas(gcf, [p2s '/ECoG_PCA_cumvar.png']);

% % Plot SC32 data projected onto the first 2 principal components
% figure; hold on;
% plot(SC32_pca.score(:,1), SC32_pca.score(:,2), 'k.', 'MarkerSize', 10);
% line([0 SC32_pca.latent(1)], [0 0], 'color', 'red', 'LineStyle', '--');
% hold off;
% xlabel('1', 'FontSize', 14); ylabel('2', 'FontSize', 14); 
% xlim([-140 50]); ylim([-50 60]);
% title('SC32 - 2 PC', 'FontSize', 16)
% saveas(gcf, [p2s '/SC32_PCA2.png']);
% 
% % Plot SC32 data projected onto the first 3 principal components
% figure; hold on;
% plot3(SC32_pca.score(:,1),SC32_pca.score(:,2),SC32_pca.score(:,3), 'k.', 'MarkerSize', 10);
% line([0 SC32_pca.latent(1)], [0 0], [0,0], 'color', 'red', 'LineStyle', '--');
% line([0 0], [0 SC32_pca.latent(2)], [0,0], 'color', 'red', 'LineStyle', '--');
% hold off;
% xlabel('1', 'FontSize', 14); ylabel('2', 'FontSize', 14); zlabel('3', 'FontSize', 14); 
% xlim([-140 50]); ylim([-50 60]); zlim([-50 50]);
% title('SC32 - 3 PC', 'FontSize', 16);
% saveas(gcf, [p2s '/SC32_PCA3.png']);
% 
% Plot SC32 variance distribution across latent dimensions
figure; 
plot(100*cumsum(SC32_pca.var./sum(SC32_pca.var)), 'k', 'LineWidth', 1);
ylim([50 100]); ylabel('Cumulative Variance Explained [%]', 'FontSize', 14)
xlim([1 15]); xlabel('Latent Dimension Number', 'FontSize', 14);
saveas(gcf, [p2s '/SC32_PCA_cumvar.png']);
%% Factor Analysis

% Use data with NaN values omitted
ECOG_FA.lfp = ECOG_pca.lfp;
ECOG_FA.lfpz = ECOG_pca.lfpz;
SC32_FA.lfp = SC32_pca.lfp;
SC32_FA.lfpz = SC32_pca.lfpz;

% ECOG_FA.lambda:   MLE of factor loadings
% ECOG_FA.psi:      MLE of specific vaiances
% ECOG_FA.T:        m x m factor loadings rotation matrix T
% ECOG_FA.stats:    Struct containing information relating to the null hypothesis
%   stats.loglike:  Maximized log-likelihood value
%   stats.dfe:      Error degrees of freedom
%   stats.chisq:    Approx. chi-squared stat for the null hypothesis
%   stats.p:        Right tail significance level for the null hypothesis
rank(ECOG_FA.lfp');
[ECOG_FA.lambda, ECOG_FA.psi, ECOG_FA.T, ECOG_FA.stats] = factoran(ECOG_FA.lfp', 3);

% Plot 
figure; hold on;
% line([0 1], [0 0], [0 0], 'color', 'red', 'LineStyle', '--');
% line([0 0], [0 1], [0 0], 'color', 'red', 'LineStyle', '--');
% line([0 0], [0 0], [0 1], 'color', 'red', 'LineStyle', '--');
plot(ECOG_FA.lambda(:,1), ECOG_FA.lambda(:,2), 'k.', 'MarkerSize', 10);
% plot3(ECOG_FA.lambda(:,1), ECOG_FA.lambda(:,2), ECOG_FA.lambda(:,3), 'k.', 'MarkerSize', 10);
xlabel('1'), ylabel('2'),% zlabel('3');




















