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

%% PCA

% Within Trial (2D)
trialID = 1;    % Trial to analyze
lfpdat2D = squeeze(trLfpData(trialID, :,:));

% Trial Averaging
lfpdat = squeeze(mean(trLfpData, 1));

% Perform PCA
[ECOG_pca.coeff, ECOG_pca.score, ECOG_pca.latent, ECOG_pca.tsquared, ECOG_pca.explained, ECOG_pca.mu] = pca(lfpdat(trialInfo.ECOG_indices, :));
[SC32_pca.coeff, SC32_pca.score, SC32_pca.latent, SC32_pca.tsquared, SC32_pca.explained, SC32_pca.mu] = pca(lfpdat(trialInfo.SC32_indices, :));

% Plot ECoG data projected onto the first 2 principal components
figure; hold on;
plot(ECOG_pca.score(:,1), ECOG_pca.score(:,2), 'k.', 'MarkerSize', 10);
line([0 ECOG_pca.latent(1)], [0 0], 'color', 'red', 'LineStyle', '--');
hold off;
xlabel('1'); ylabel('2'); xlim([-600 400]); ylim([-250 250]);
title('ECoG - 2')

% Plot ECoG data projected onto the first 3 principal components
figure; hold on;
plot3(ECOG_pca.score(:,1),ECOG_pca.score(:,2),ECOG_pca.score(:,3), 'k.', 'MarkerSize', 10);
line([0 ECOG_pca.latent(1)], [0 0], [0,0], 'color', 'red', 'LineStyle', '--');
line([0 0], [0 ECOG_pca.latent(2)], [0,0], 'color', 'red', 'LineStyle', '--');
hold off;
xlabel('1'); ylabel('2'); zlabel('3'); xlim([-600 400]); ylim([-250 250]); zlim([-200 300]);
title('ECoG - 3');

% Plot ECoG data projected onto the first 2 principal components
figure; hold on;
plot(SC32_pca.score(:,1), SC32_pca.score(:,2), 'k.', 'MarkerSize', 10);
line([0 SC32_pca.latent(1)], [0 0], 'color', 'red', 'LineStyle', '--');
hold off;
xlabel('1'); ylabel('2'); 
xlim([-2000 2000]); ylim([-2000 2000]);
title('SC32 - 2')

% Plot SC32 data projected onto the first 3 principal components
figure; hold on;
plot3(SC32_pca.score(:,1),SC32_pca.score(:,2),SC32_pca.score(:,3), 'k.', 'MarkerSize', 10);
line([0 SC32_pca.latent(1)], [0 0], [0,0], 'color', 'red', 'LineStyle', '--');
line([0 0], [0 SC32_pca.latent(2)], [0,0], 'color', 'red', 'LineStyle', '--');
hold off;
xlabel('1'); ylabel('2'); zlabel('3'); 
xlim([-2000 2000]); ylim([-2000 2000]); zlim([-2000 2000]);
title('SC32 - 3');

%% Factor Analysis

% Remove NaN
lfpdat_cut = reshape(lfpdat(~isnan(lfpdat)),[], size(lfpdat,2));

% lambda: factor loadings
% psi: maximum likelihood estimates of the speciifc variances
% T: mxm factor loadings rotation matrix
% stats: struct with information relating to the null hypothesis
[ECOG_FA.lambda, ECOG_FA.psi, ECOG_FA.T, ECOG_FA.stats] = factoran(lfpdat_cut(trialInfo.ECOG_indices(1):trialInfo.ECOG_indices(end)-length(trialInfo.badECoGs), :)', 3);

%% Plot 
figure; hold on;
plot3(ECOG_FA.lambda(:,1), ECOG_FA.lambda(:,2), ECOG_FA.lambda(:,3), 'k.', 'MarkerSize', 10);
xlabel('1'), ylabel('2'), zlabel('3');




















