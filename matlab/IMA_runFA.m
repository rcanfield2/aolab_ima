%% Run FA Basic
close all;
clear all;
clc;

%% Define parameters
global MONKEYDIR
MONKEYDIR = 'E:\aolab\data\centerOut_ECOG';
plotinfo.showplots = true;  % Should plots be shown
plotinfo.saveplots = false; % Should plots be saved
plotinfo.p2s = 'E:/aolab/data/centerOut_ECOG/figures';  % Path to save
wcECOG = [100]; % Butterworth filter frequency cutoff for ECOG data
wcSC32 = 25;    % Butterworth filter frequency cutoff for SC32 data
bwOrder = 4;    % Butterwoth filter order
avgdTrials = 1; % Which trials to average. 
                    % [trialidx] to look at a specific trial
                    % [startidx:endidx] to define a range
                    % 'all' to average all trials. 

%% Analysis 
% Load Data
[trLfpData, trialInfo] = IMA_loadData(MONKEYDIR);
trialInfo.avgdTrials = avgdTrials;

% Preprocess ECOG
plotinfo.idx = 100;         % Channel index to plot
plotinfo.ID = 'ECOG';       % String ID for preprocessing run
ecogLfpPrePCA = IMA_preprocess(trLfpData(:,trialInfo.ECOG_indices,:),trialInfo.badECoGs, wcECOG, bwOrder, trialInfo, plotinfo);

% Preprocess SC32
plotinfo.idx = 7;         % Channel index to plot
plotinfo.ID = 'SC32';       % String ID for preprocessing run
sc32LfpPrePCA = IMA_preprocess(trLfpData(:,trialInfo.SC32_indices-trialInfo.ECoG_offset,:), trialInfo.badE(4:end)-trialInfo.ECoG_offset, wcSC32, bwOrder, trialInfo, plotinfo);

% Run FA on ECOG
FAparams.m = 1:20;
for ii = FAparams.m
    ecogFA = IMA_FA(ecogLfpPrePCA', trialInfo.goodECoGs, ii);
    ll(ii) = ecogFA.stats.loglike;
    disp(num2str(ii));
end

figure; plot(ll)