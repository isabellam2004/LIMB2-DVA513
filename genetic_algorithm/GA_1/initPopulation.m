function population = initPopulation(popSize)
% initPopulation - Generates the initial population for the LIMB GA
%
% INPUT:
%   popSize - number of chromosomes in the population (e.g. 100)
%
% OUTPUT:
%   population - matrix of size [popSize x 70]
%                each row is one chromosome (one candidate user profile)
%
% CHROMOSOME STRUCTURE (70 genes):
%   Genes 1-20:  Arm joints (sh_flex, sh_abd, el_flex, wr_pron)
%   Genes 21-70: Finger joints (thumb, index, middle, ring, pinky)
%   Per joint, gene order: [max_angle, start_angle, smoothness, time_to_peak_min, time_to_peak_max]
%
% Author: Alina
% Project: LIMB - Personalised Motor Rehabilitation

% -------------------------------------------------------------------------
% BOUNDS DEFINITION
% Each row = [min, max] for that gene
% Order matches chromosome gene index exactly
% -------------------------------------------------------------------------

bounds = [
    % --- ARM JOINTS ---
    % sh_flex (shoulder flexion)
    -60,  180;   % gene 1:  sh_flex · max_angle
    -60,   60;   % gene 2:  sh_flex · start_angle
      0,    1;   % gene 3:  sh_flex · smoothness
    0.3,    7;   % gene 4:  sh_flex · time_to_peak_min
    0.3,    7;   % gene 5:  sh_flex · time_to_peak_max

    % sh_abd (shoulder abduction)
      0,  180;   % gene 6:  sh_abd · max_angle
      0,   60;   % gene 7:  sh_abd · start_angle
      0,    1;   % gene 8:  sh_abd · smoothness
    0.3,    7;   % gene 9:  sh_abd · time_to_peak_min
    0.3,    7;   % gene 10: sh_abd · time_to_peak_max

    % el_flex (elbow flexion)
      0,  160;   % gene 11: el_flex · max_angle
      0,  160;   % gene 12: el_flex · start_angle  (patient rests with arm already bent)
      0,    1;   % gene 13: el_flex · smoothness
    0.3,    7;   % gene 14: el_flex · time_to_peak_min
    0.3,    7;   % gene 15: el_flex · time_to_peak_max

    % wr_pron (wrist pronation) — EXCLUDED from fitness scoring (unreliable camera signal)
      0,  180;   % gene 16: wr_pron · max_angle
      0,  180;   % gene 17: wr_pron · start_angle
      0,    1;   % gene 18: wr_pron · smoothness
    0.3,    7;   % gene 19: wr_pron · time_to_peak_min
    0.3,    7;   % gene 20: wr_pron · time_to_peak_max

    % --- FINGER JOINTS ---
    % thumb_MCP
      0,   70;   % gene 21: thumb_MCP · max_angle
      0,   40;   % gene 22: thumb_MCP · start_angle
      0,    1;   % gene 23: thumb_MCP · smoothness
    0.2,    7;   % gene 24: thumb_MCP · time_to_peak_min
    0.2,    7;   % gene 25: thumb_MCP · time_to_peak_max

    % thumb_PIP
      0,   90;   % gene 26: thumb_PIP · max_angle
      0,   40;   % gene 27: thumb_PIP · start_angle
      0,    1;   % gene 28: thumb_PIP · smoothness
    0.2,    7;   % gene 29: thumb_PIP · time_to_peak_min
    0.2,    7;   % gene 30: thumb_PIP · time_to_peak_max

    % index_MCP
      0,  145;   % gene 31: index_MCP · max_angle
      0,  145;   % gene 32: index_MCP · start_angle  (rests ~110-120 in natural posture)
      0,    1;   % gene 33: index_MCP · smoothness
    0.2,    7;   % gene 34: index_MCP · time_to_peak_min
    0.2,    7;   % gene 35: index_MCP · time_to_peak_max

    % index_PIP
      0,  120;   % gene 36: index_PIP · max_angle
      0,   60;   % gene 37: index_PIP · start_angle
      0,    1;   % gene 38: index_PIP · smoothness
    0.2,    7;   % gene 39: index_PIP · time_to_peak_min
    0.2,    7;   % gene 40: index_PIP · time_to_peak_max

    % middle_MCP
      0,  145;   % gene 41: middle_MCP · max_angle
      0,  145;   % gene 42: middle_MCP · start_angle
      0,    1;   % gene 43: middle_MCP · smoothness
    0.2,    7;   % gene 44: middle_MCP · time_to_peak_min
    0.2,    7;   % gene 45: middle_MCP · time_to_peak_max

    % middle_PIP
      0,  120;   % gene 46: middle_PIP · max_angle
      0,   60;   % gene 47: middle_PIP · start_angle
      0,    1;   % gene 48: middle_PIP · smoothness
    0.2,    7;   % gene 49: middle_PIP · time_to_peak_min
    0.2,    7;   % gene 50: middle_PIP · time_to_peak_max

    % ring_MCP
      0,  145;   % gene 51: ring_MCP · max_angle
      0,  145;   % gene 52: ring_MCP · start_angle
      0,    1;   % gene 53: ring_MCP · smoothness
    0.2,    7;   % gene 54: ring_MCP · time_to_peak_min
    0.2,    7;   % gene 55: ring_MCP · time_to_peak_max

    % ring_PIP
      0,  120;   % gene 56: ring_PIP · max_angle
      0,   60;   % gene 57: ring_PIP · start_angle
      0,    1;   % gene 58: ring_PIP · smoothness
    0.2,    7;   % gene 59: ring_PIP · time_to_peak_min
    0.2,    7;   % gene 60: ring_PIP · time_to_peak_max

    % pinky_MCP
      0,  145;   % gene 61: pinky_MCP · max_angle
      0,  145;   % gene 62: pinky_MCP · start_angle
      0,    1;   % gene 63: pinky_MCP · smoothness
    0.2,    7;   % gene 64: pinky_MCP · time_to_peak_min
    0.2,    7;   % gene 65: pinky_MCP · time_to_peak_max

    % pinky_PIP
      0,  120;   % gene 66: pinky_PIP · max_angle
      0,   60;   % gene 67: pinky_PIP · start_angle
      0,    1;   % gene 68: pinky_PIP · smoothness
    0.2,    7;   % gene 69: pinky_PIP · time_to_peak_min
    0.2,    7;   % gene 70: pinky_PIP · time_to_peak_max
];

% -------------------------------------------------------------------------
% POPULATION GENERATION
% For each gene, generate popSize random values uniformly between [min, max]
% -------------------------------------------------------------------------

numGenes = size(bounds, 1);  % should be 70
population = zeros(popSize, numGenes);

for g = 1:numGenes
    lb = bounds(g, 1);  % lower bound for this gene
    ub = bounds(g, 2);  % upper bound for this gene
    population(:, g) = lb + (ub - lb) * rand(popSize, 1);
end

% -------------------------------------------------------------------------
% CONSTRAINT: time_to_peak_min <= time_to_peak_max
% Gene pairs: (4,5), (9,10), (14,15), (19,20), then every (24,25)...(69,70)
% -------------------------------------------------------------------------

timePeakMinGenes = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69];

for i = 1:length(timePeakMinGenes)
    minIdx = timePeakMinGenes(i);
    maxIdx = minIdx + 1;
    
    % Where min > max, swap them
    needSwap = population(:, minIdx) > population(:, maxIdx);
    temp = population(needSwap, minIdx);
    population(needSwap, minIdx) = population(needSwap, maxIdx);
    population(needSwap, maxIdx) = temp;
end

% -------------------------------------------------------------------------
% QUICK SANITY CHECK (prints to console)
% -------------------------------------------------------------------------

fprintf('Population initialised: %d chromosomes x %d genes\n', popSize, numGenes);
fprintf('Gene ranges check (first chromosome):\n');
fprintf('  sh_flex max_angle   : %.2f  (bounds: -60 to 180)\n', population(1, 1));
fprintf('  el_flex max_angle   : %.2f  (bounds: 0 to 160)\n',   population(1, 11));
fprintf('  smoothness sample   : %.3f  (bounds: 0 to 1)\n',     population(1, 3));
fprintf('  time_to_peak check  : min=%.2f <= max=%.2f\n',       population(1, 4), population(1, 5));

end
