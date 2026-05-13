function parents = tournamentSelection(population, fitnessScores, tournamentSize)
% tournamentSelection - Selects parents via tournament selection
%
% INPUT:
%   population     - matrix [popSize x 70] of current chromosomes
%   fitnessScores  - vector [popSize x 1] of fitness values (lower = better)
%   tournamentSize - how many chromosomes compete in each tournament (e.g. 3)
%
% OUTPUT:
%   parents        - matrix [popSize x 70] of selected parents
%                    same size as population, ready for crossover
%
% HOW IT WORKS:
%   To fill one parent slot:
%     1. Pick tournamentSize random chromosomes from the population
%     2. Compare their fitness scores
%     3. The one with the LOWEST fitness (= best) wins and becomes a parent
%   Repeat popSize times to fill the entire parents matrix.
%   The same chromosome can win multiple tournaments (intentional).
%
% WHY TOURNAMENT SELECTION:
%   - Simple and effective
%   - tournamentSize controls selection pressure:
%       size=2 → low pressure, more diversity
%       size=5 → high pressure, best chromosomes dominate faster
%       size=3 → good balance (default for LIMB)
%
% Author: Alina
% Project: LIMB - Personalised Motor Rehabilitation

[popSize, numGenes] = size(population);
parents = zeros(popSize, numGenes);

% -------------------------------------------------------------------------
% RUN popSize TOURNAMENTS — one winner per slot
% -------------------------------------------------------------------------

for i = 1:popSize

    % Step 1: Pick tournamentSize random indices (no replacement)
    candidates = randperm(popSize, tournamentSize);

    % Step 2: Get their fitness scores
    candidateFitness = fitnessScores(candidates);

    % Step 3: Find the winner (lowest fitness = best)
    [~, winnerLocalIdx] = min(candidateFitness);
    winnerIdx = candidates(winnerLocalIdx);

    % Step 4: Copy winner into parents
    parents(i, :) = population(winnerIdx, :);

end

end