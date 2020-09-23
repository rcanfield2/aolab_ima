function [] = IMA_plotPC(plotinfo, varargin)
%% 
% Plot data projects on the first principal componants (PC). The number of
% principal components is specified by the number of pc vector input
% arguements.
%
% Inputs
%   p: [int] - Number of PC componants to plot (Between 1 and 3)
%   plotinfo: [struct] - Plot information
%   
%   Optional (in order from pc1 to pc3):
%       pc1: [nchannel] - Data projected onto PC1
%       pc2: [nchannel] - Data projected onto PC2
%       pc3: [nchannel] - Data projected onto PC3
%
% Outputs:
%   Plots one of the following based on length(varargin):
%       - PC1 vs. Observation
%       - PC1 vs. PC2
%       - PC1 vs. PC2 vs. PC3
%%

switch length(varargin)
    case 1
        figure; hold on;
        plot(varargin{1}, 'k.', 'MarkerSize', 10);
        hold off;
        xlabel('Channel ID', 'FontSize', 14);
        ylabel('Varaince in PC1', 'FontSize', 14);
        title([plotinfo.ID ' - 1 PC'], 'FontSize', 16)
        if plotinfo.saveplots
            saveas(gcf, [plotinfo.p2s '/' plotinfo.ID '_PCA1.png']);
        end
    case 2
        figure; hold on;
        plot(varargin{1}, varargin{2}, 'k.', 'MarkerSize', 10);
        hold off;
        xlabel('1', 'FontSize', 14); ylabel('2', 'FontSize', 14);
        title([plotinfo.ID ' - 2 PC'], 'FontSize', 16)
        if plotinfo.saveplots
            saveas(gcf, [plotinfo.p2s '/' plotinfo.ID '_PCA2.png']);
        end
    case 3
        figure; hold on;
        plot3(varargin{1}, varargin{2}, varargin{3}, 'k.', 'MarkerSize', 10);
        hold off;
        xlabel('1', 'FontSize', 14); ylabel('2', 'FontSize', 14); 
        zlabel('3', 'FontSize', 14)
        title([plotinfo.ID ' - 3 PC'], 'FontSize', 16)
        view([-45 30]);
        if plotinfo.saveplots
            saveas(gcf, [plotinfo.p2s '/' plotinfo.ID '_PCA3.png']);
        end
    otherwise
        error('Too many optional arguements');
end

