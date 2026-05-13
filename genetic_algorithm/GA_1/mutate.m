function offspring = mutate(offspring, mutationRate)
% mutate - Applies Gaussian mutation to a population of chromosomes
%
% INPUT:
%   offspring    - matrix [popSize x 70] of chromosomes to mutate
%   mutationRate - probability of mutating each individual gene (e.g. 0.02 = 2%)
%
% OUTPUT:
%   offspring    - same matrix with some genes randomly nudged
%
% RULES:
%   - Each gene is mutated independently with probability = mutationRate
%   - Mutation = Gaussian noise scaled to the gene's range
%   - Mutated values are clipped to stay within bounds
%   - time_to_peak_min <= time_to_peak_max is enforced after mutation
%
% Author: Alina
% Project: LIMB - Personalised Motor Rehabilitation

% -------------------------------------------------------------------------
% BOUNDS — same as initPopulation.m (must stay in sync)
% -------------------------------------------------------------------------

bounds = [
    -30,  180;   % gene 1:  sh_flex · max_angle
    -30,   30;   % gene 2:  sh_flex · start_angle
      0,    1;   % gene 3:  sh_flex · smoothness
    0.3,    2;   % gene 4:  sh_flex · time_to_peak_min
    0.3,    5;   % gene 5:  sh_flex · time_to_peak_max
      0,   90;   % gene 6:  sh_abd · max_angle
      0,   20;   % gene 7:  sh_abd · start_angle
      0,    1;   % gene 8:  sh_abd · smoothness
    0.3,    2;   % gene 9:  sh_abd · time_to_peak_min
    0.3,    5;   % gene 10: sh_abd · time_to_peak_max
      0,  145;   % gene 11: el_flex · max_angle
      0,   20;   % gene 12: el_flex · start_angle
      0,    1;   % gene 13: el_flex · smoothness
    0.3,    2;   % gene 14: el_flex · time_to_peak_min
    0.3,    5;   % gene 15: el_flex · time_to_peak_max
      0,   90;   % gene 16: wr_pron · max_angle
      0,   30;   % gene 17: wr_pron · start_angle
      0,    1;   % gene 18: wr_pron · smoothness
    0.3,    2;   % gene 19: wr_pron · time_to_peak_min
    0.3,    5;   % gene 20: wr_pron · time_to_peak_max
      0,   70;   % gene 21: thumb_MCP · max_angle
      0,   15;   % gene 22: thumb_MCP · start_angle
      0,    1;   % gene 23: thumb_MCP · smoothness
    0.2,    2;   % gene 24: thumb_MCP · time_to_peak_min
    0.2,    5;   % gene 25: thumb_MCP · time_to_peak_max
      0,   80;   % gene 26: thumb_PIP · max_angle
      0,   15;   % gene 27: thumb_PIP · start_angle
      0,    1;   % gene 28: thumb_PIP · smoothness
    0.2,    2;   % gene 29: thumb_PIP · time_to_peak_min
    0.2,    5;   % gene 30: thumb_PIP · time_to_peak_max
      0,   90;   % gene 31: index_MCP · max_angle
      0,   15;   % gene 32: index_MCP · start_angle
      0,    1;   % gene 33: index_MCP · smoothness
    0.2,    2;   % gene 34: index_MCP · time_to_peak_min
    0.2,    5;   % gene 35: index_MCP · time_to_peak_max
      0,  100;   % gene 36: index_PIP · max_angle
      0,   15;   % gene 37: index_PIP · start_angle
      0,    1;   % gene 38: index_PIP · smoothness
    0.2,    2;   % gene 39: index_PIP · time_to_peak_min
    0.2,    5;   % gene 40: index_PIP · time_to_peak_max
      0,   90;   % gene 41: middle_MCP · max_angle
      0,   15;   % gene 42: middle_MCP · start_angle
      0,    1;   % gene 43: middle_MCP · smoothness
    0.2,    2;   % gene 44: middle_MCP · time_to_peak_min
    0.2,    5;   % gene 45: middle_MCP · time_to_peak_max
      0,  100;   % gene 46: middle_PIP · max_angle
      0,   15;   % gene 47: middle_PIP · start_angle
      0,    1;   % gene 48: middle_PIP · smoothness
    0.2,    2;   % gene 49: middle_PIP · time_to_peak_min
    0.2,    5;   % gene 50: middle_PIP · time_to_peak_max
      0,   90;   % gene 51: ring_MCP · max_angle
      0,   15;   % gene 52: ring_MCP · start_angle
      0,    1;   % gene 53: ring_MCP · smoothness
    0.2,    2;   % gene 54: ring_MCP · time_to_peak_min
    0.2,    5;   % gene 55: ring_MCP · time_to_peak_max
      0,  100;   % gene 56: ring_PIP · max_angle
      0,   15;   % gene 57: ring_PIP · start_angle
      0,    1;   % gene 58: ring_PIP · smoothness
    0.2,    2;   % gene 59: ring_PIP · time_to_peak_min
    0.2,    5;   % gene 60: ring_PIP · time_to_peak_max
      0,   90;   % gene 61: pinky_MCP · max_angle
      0,   15;   % gene 62: pinky_MCP · start_angle
      0,    1;   % gene 63: pinky_MCP · smoothness
    0.2,    2;   % gene 64: pinky_MCP · time_to_peak_min
    0.2,    5;   % gene 65: pinky_MCP · time_to_peak_max
      0,  100;   % gene 66: pinky_PIP · max_angle
      0,   15;   % gene 67: pinky_PIP · start_angle
      0,    1;   % gene 68: pinky_PIP · smoothness
    0.2,    2;   % gene 69: pinky_PIP · time_to_peak_min
    0.2,    5;   % gene 70: pinky_PIP · time_to_peak_max
];

% -------------------------------------------------------------------------
% MUTATION
% -------------------------------------------------------------------------

[popSize, numGenes] = size(offspring);

for i = 1:popSize
    for g = 1:numGenes

        if rand() < mutationRate

            lb    = bounds(g, 1);
            ub    = bounds(g, 2);
            range = ub - lb;

            % Gaussian noise scaled to 5% of gene range
            % Small enough to nudge, large enough to explore
            noise = randn() * (0.05 * range);

            % Apply mutation and clip to bounds
            offspring(i, g) = max(lb, min(ub, offspring(i, g) + noise));
        end

    end
end

% -------------------------------------------------------------------------
% ENFORCE time_to_peak_min <= time_to_peak_max after mutation
% Gene index pairs: (4,5), (9,10), (14,15), (19,20), then fingers
% -------------------------------------------------------------------------

timePeakMinGenes = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69];

for i = 1:length(timePeakMinGenes)
    minIdx = timePeakMinGenes(i);
    maxIdx = minIdx + 1;

    % Where min > max after mutation, swap them
    needSwap = offspring(:, minIdx) > offspring(:, maxIdx);
    temp = offspring(needSwap, minIdx);
    offspring(needSwap, minIdx) = offspring(needSwap, maxIdx);
    offspring(needSwap, maxIdx) = temp;
end

end