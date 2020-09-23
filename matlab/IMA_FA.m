function [FAout] = IMA_FA(lfpdat, goodE, m)
%%
% This function computes the maximum likelihood estimate (MLE) of the
% factor loadings matrix in the factor analysis model given neural data
% over m common factors
%   x = mu + lambda*f + e
%       x: vector of observed variables (length m)
%       mu: constant vector of means (length m)
%       lambda: constant d x m matrix of factor loadings
%       f: vecotr of independent, standardized common factors (length m)
%       e: vector of independent specific factors (length m)
%
% Inputs;
%   lfpin: [ntime, nchannel] - Preprocessed data where each row corresponds
%       to an observation and each column correspond to a variable.
%   goodE: [ngoodchannel] - Vector containing the indices of good channels
%   m: [int] - Number of common factors
%
% Outputs
%   FAout: [struct] - Contains FA results
%       FAout.lambda: [nchannel, m] Factor loadings for m common factors
%       FAout.psi: [nchannel] MLE of specfici variances
%       FAout.T: [m x m] factor loadings rotation matrix
%       FAout.stats: [struct] Information relating to the null hypothesis
%           that the number of common factors is m
%       FAout.F: [ntime, m] Predictions of the common factors
%%

% Remove bad electrode NaNs
if size(lfpdat, 1) > size(lfpdat, 2)
    lfpdat = lfpdat(:, goodE);
else
    lfpdat = lfpdat(goodE, :);
end

% Perform factor analysis
[FAout.lambda, FAout.psi, FAout.T, FAout.stats, FAout.F] = factoran(lfpdat, m);