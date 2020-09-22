function [ecogPCA] = IMA_PCA(lfpin, goodE, zscoreParam)
%% 
% This function zscores preprocessed LFP data and computes the PCA using
% the SVD
%
% Inputs:
%   lfpin: [ntime, nchannel] - Preprocessed data where each row corresponds
%       to an observation and each column correspond to a variable.
%   zscoreParam: [struct] - Parameters used to zscore data
%
% Outputs:
%   ecogPCA: [struct] - Contains PCA results
%       ecogPCA.coeff: - loadings
%       ecogPCA.score: - Representation of lfpin in the principal component
%           space
%       ecogPCA.latent: - Principal component (PC) variances
%       ecogPCA.tsq = Hotelling's T-squared statistic for each observation.
%       ecogPCA.explained: - Percent of total varaince explained by each PC
%       ecogPCA.mu: - Estimated mean of each variable in lfpin.
%%

% Remove bad electrode NaNs
if size(lfpin, 1) > size(lfpin, 2)
    lfpin = lfpin(:, goodE);
else
    lfpin = lfpin(goodE, :);
end

% Z-score
ecogPCA.lfpz = zscore(lfpin, zscoreParam.stdScale, zscoreParam.dim);

% Perform PCA
[ecogPCA.coeff, ecogPCA.score, ecogPCA.latent, ecogPCA.tsq, ecogPCA.explained, ecogPCA.mu] = pca(ecogPCA.lfpz, 'algorithm', 'svd');