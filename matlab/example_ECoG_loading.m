%make sure the following folders are included in matlab paths:
%   spont
%   morePackages
%   preprocessing
%   analyzeflashing

%the key inputs
%   MONKEYDIR: the data directory
%   selectedfile: target matching template

%change this to your data directory
%this assumes a directory like this
%   MONKEYDIR
%       dates in YYMMDD eg.
%           rec_nums in three digits eg. 001
%               .lfp
%               experiment.m 
%               events.m
%           mat 
%               trials.m

% load some example recordings
global MONKEYDIR selectedfile


%MONKEYDIR = 'D:\SiJia\OneDrive - UW\projects\Brain EEG\data';
MONKEYDIR = 'E:\aolab\data\centerOut_ECOG';

selectedfile = [MONKEYDIR, '\180328\posTargets_180328.mat'];
addpath(genpath([MONKEYDIR '/m/']))
addpath(genpath('E:\aolab\codes\Si Jia\multiscale_analysis-master'))

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

%the key matrices (tensors) are
% trLfpData : num_trial by  (211 ECoG electrodes + 32 intracortical
% electrodes ) by timepoints
% trial_TargetAssigned: num_trial by 1
% trialInfo: a structure of useful metadata for processing
%% looking at some example recordings
trial_i = 1;
%ECoG
ecog_electrode_i = 1;
%intracortical electrode 
sc32_electrode_i = 6;
sc32_electrode_index = trialInfo.ECoG_offset + sc32_electrode_i; 
%can also slice out either ECoG or intracortical recordings e.g.
%   trLfpData_ecog = trLfpData(:, 1:ECoG_ offset, :)
%   trLfpData_sc32 = trLfpData(:, ECoG_offset:end, :)

figure
%plot from electrode 1 during the 1st trial
plot(trialInfo.timeVec, squeeze(trLfpData(trial_i,ecog_electrode_i,:)))
xlabel('Time from movement onset(ms)')

hold on
%plot from electrode 6 during the 1st trial
plot(trialInfo.timeVec, squeeze(trLfpData(trial_i,sc32_electrode_index,:)))

xline(0)
legend(['ECoG ', num2str(ecog_electrode_i)], ...
       ['SC32 ', num2str(sc32_electrode_i)])
   
%% load the electrodes layout

%P_chamber translates coord from within drive to relative to the entire
%setup i.e. chamber coors
P_chamber_ECOG = drmap_coordTransform_microdrive2chamber(experiment,1);
P_chamber_ECOG = P_chamber_ECOG';

P_chamber_SC32 = drmap_coordTransform_microdrive2chamber(experiment,2);
P_chamber_SC32 = P_chamber_SC32';


figure
hold on
scatter(P_chamber_ECOG(:,1), P_chamber_ECOG(:,2))
scatter(P_chamber_SC32(:,1), P_chamber_SC32(:,2))
hold off
title('Clean Electrodes Near the Lowered SC32 Electrodes')
xlabel('mm')
ylabel('mm')
legend('ECoG' , 'SC32')
