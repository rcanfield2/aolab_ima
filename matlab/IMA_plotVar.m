function [] = IMA_plotVar(plotinfo, PCAresults)
%%
% This function plots the variance distribution across latent dimensions on
% a semi log plot and the cumulative variance percentage across latent
% dimensions
%
% Inputs:
%   plotinfo: [struct] - Plot information
%   PCAresults: [struct] - PCA results
%
% Outputs:
%   Plots:
%       - Variance distribution vs. latent dimension
%       - Cumulative distribution percent vs. latent dimension
%%

% Plot variance distribution
figure; 
semilogy(PCAresults.latent, ':k.', 'MarkerSize', 10, 'LineWidth', 1);
xlabel('Latent Dimension', 'FontSize', 14);
ylabel('Varaince' , 'FontSize', 14);
title([plotinfo.ID ' Variance Distribution Across Latent Dimensions']);
xlim([1, length(PCAresults.latent)]);
if plotinfo.saveplots
    saveas(gcf, [plotinfo.p2s '/' plotinfo.ID '_PCA_cumvar.png']);
end

% Plot cumulative variance 
figure; 
plot(100*cumsum(PCAresults.latent/sum(PCAresults.latent)), ':k.', ...
    'MarkerSize', 10, 'Linewidth', 1);
xlabel('Latent Dimension', 'FontSize', 14);
ylabel('Percent of Total Variance', 'FontSize', 14);
title([plotinfo.ID ' Cumulative Variance Distribution']);
xlim([1, ceil(length(PCAresults.latent)/3)]);
if plotinfo.saveplots
    saveas(gcf, [plotinfo.p2s '/' plotinfo.ID '_PCA_cumvar.png']);
end
