%% Run PCA Basic
close all;
clear all;
clc

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

% Run PCA on ECOG
zscoreParam.stdScale = 0;
zscoreParam.dim = 'all';
ecogPCA = IMA_PCA(ecogLfpPrePCA', trialInfo.goodECoGs, zscoreParam);

% Run PCA on SC32
zscoreParam.stdScale = 0;
zscoreParam.dim = 'all';
sc32PCA = IMA_PCA(sc32LfpPrePCA', trialInfo.goodSC32, zscoreParam);

%% Plot

% Plot ECOG projections onto PC Axis
plotinfo.ID = 'ECOG';       % String ID for preprocessing run
IMA_plotPC(plotinfo, ecogPCA.score(:,1));
IMA_plotPC(plotinfo, ecogPCA.score(:,1), ecogPCA.score(:,2));
IMA_plotPC(plotinfo, ecogPCA.score(:,1), ecogPCA.score(:,2), ecogPCA.score(:,3));

% Plot ECOG Variance
plotinfo.ID = 'ECOG';
IMA_plotVar(plotinfo, ecogPCA);

% Plot SC32 projections onto PC Axis
plotinfo.ID = 'SC32';       % String ID for preprocessing run
IMA_plotPC(plotinfo, sc32PCA.score(:,1));
IMA_plotPC(plotinfo, sc32PCA.score(:,1), sc32PCA.score(:,2));
IMA_plotPC(plotinfo, sc32PCA.score(:,1), sc32PCA.score(:,2), sc32PCA.score(:,3));

% Plot SC32 Variance
plotinfo.ID = 'SC32';
IMA_plotVar(plotinfo, sc32PCA);