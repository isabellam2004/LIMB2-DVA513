function [bestChromosome, bestFitness] = runGA(gaInputPath, patientId, patientName)
% runGA - Main Genetic Algorithm loop for LIMB personalised kinematic profiling
%
% INPUT:
%   gaInputPath  - path to the GA input JSON produced by jsonSetup.py
%                  (e.g. 'patient_4_ga_input.json')
%                  Pass [] to run in test mode with random fitness.
%   patientId    - integer patient ID (e.g. 4)
%   patientName  - string patient name (e.g. 'John Doe')
%
% OUTPUT:
%   bestChromosome - 1x70 vector, the optimal kinematic profile for this user
%   Writes:        patient_<id>_profile.json to the current folder
%
% USAGE:
%   bestChromosome = runGA([], 0, 'Test');                              % test mode
%   bestChromosome = runGA('patient_4_ga_input.json', 4, 'John Doe');  % real mode
%
% Author: Alina
% Project: LIMB - Personalised Motor Rehabilitation

% -------------------------------------------------------------------------
% GA PARAMETERS — tweak these to improve performance
% -------------------------------------------------------------------------

popSize        = 200;    % number of chromosomes in population
maxGenerations = 500;    % hard stop: max number of generations
stagnationLimit = 50;    % soft stop: stop if no improvement for N generations
crossoverRate  = 0.8;    % probability that two parents do crossover (80%)
mutationRate   = 0.02;   % probability of mutating each gene (2%)
tournamentSize = 10;      % number of chromosomes compared in each tournament

% -------------------------------------------------------------------------
% STEP 1 — Initialise population
% -------------------------------------------------------------------------

fprintf('=== LIMB GA Starting ===\n');
fprintf('Population: %d | Max generations: %d | Stagnation limit: %d\n', ...
    popSize, maxGenerations, stagnationLimit);

population = initPopulation(popSize);

% -------------------------------------------------------------------------
% STEP 2 — Evaluate initial fitness
% -------------------------------------------------------------------------

fitnessScores = evaluateFitness(population, gaInputPath);

% Track best solution found so far
[bestFitness, bestIdx] = min(fitnessScores);  % min because fitness = error (lower is better)
bestChromosome = population(bestIdx, :);

fprintf('Generation 0 | Best fitness: %.4f\n', bestFitness);

% -------------------------------------------------------------------------
% STEP 3 — Main evolution loop
% -------------------------------------------------------------------------

stagnationCount = 0;
fitnessHistory  = zeros(1, maxGenerations);

for gen = 1:maxGenerations

    % --- SELECTION ---
    % Pick parents using tournament selection
    parents = tournamentSelection(population, fitnessScores, tournamentSize);

    % --- CROSSOVER ---
    % Combine pairs of parents to produce offspring
    offspring = crossover(parents, crossoverRate);

    % --- MUTATION ---
    % Randomly nudge some genes in the offspring
    offspring = mutate(offspring, mutationRate);

    % --- EVALUATE FITNESS of new offspring ---
    offspringFitness = evaluateFitness(offspring, gaInputPath);

    % --- ELITISM — preserve best chromosome into next generation ---
    % Replace the worst offspring with the best chromosome found so far.
    % This guarantees fitness never spikes back up between generations.
    [~, worstOffspringIdx] = max(offspringFitness);
    offspring(worstOffspringIdx, :) = bestChromosome;
    offspringFitness(worstOffspringIdx) = bestFitness;

    % --- REPLACE population with offspring ---
    population    = offspring;
    fitnessScores = offspringFitness;

    % --- TRACK BEST ---
    [genBestFitness, genBestIdx] = min(fitnessScores);
    fitnessHistory(gen) = genBestFitness;

    if genBestFitness < bestFitness
        bestFitness    = genBestFitness;
        bestChromosome = population(genBestIdx, :);
        stagnationCount = 0;  % reset stagnation counter
    else
        stagnationCount = stagnationCount + 1;
    end

    % --- PRINT PROGRESS every 50 generations ---
    if mod(gen, 50) == 0
        fprintf('Generation %d | Best fitness: %.4f | Stagnation: %d/%d\n', ...
            gen, bestFitness, stagnationCount, stagnationLimit);
    end

    % --- STOPPING CRITERIA: stagnation ---
    if stagnationCount >= stagnationLimit
        fprintf('\nStopped early at generation %d — no improvement for %d generations.\n', ...
            gen, stagnationLimit);
        break;
    end

end

% -------------------------------------------------------------------------
% STEP 4 — Results
% -------------------------------------------------------------------------

fprintf('\n=== GA Finished ===\n');
fprintf('Best fitness achieved: %.4f\n', bestFitness);
fprintf('Best chromosome (first 10 genes):\n');
disp(bestChromosome(1:10));

% Plot fitness over generations (Diana uses this for convergence plots too)
figure;
plot(fitnessHistory(1:gen));
xlabel('Generation');
ylabel('Best Fitness (lower = better)');
title('LIMB GA Convergence');
grid on;

% -------------------------------------------------------------------------
% STEP 5 — Profile saving is handled by runAllMovements.m
% -------------------------------------------------------------------------
% runGA returns bestChromosome and bestFitness to the orchestrator,
% which collects all 3 movement results and saves the profile in one shot.

end


% =========================================================================
% LOCAL HELPER FUNCTIONS
% (these will be replaced by Alina's full implementations in separate files)
% =========================================================================

function fitnessScores = evaluateFitness(population, gaInputPath)
% Connects Alina's GA engine to the fitness function.
%
%   Test mode (no data):  bestChromosome = runGA([]);
%   Real mode (with data): bestChromosome = runGA('patient_4_ga_input.json');

    if isempty(gaInputPath)
        % Test mode — random fitness so the GA loop can be tested without data
        fitnessScores = rand(size(population, 1), 1);
    else
        % Real mode — call the fitness function with the JSON path
        fitnessScores = fitnessFunction(population, gaInputPath);
    end
end


function parents = tournamentSelection(population, fitnessScores, tournamentSize)
% PLACEHOLDER — full implementation goes in tournamentSelection.m
% Selects popSize parents via tournament selection

    popSize  = size(population, 1);
    parents  = zeros(size(population));

    for i = 1:popSize
        % Pick tournamentSize random candidates
        candidates = randperm(popSize, tournamentSize);
        % Winner = lowest fitness (lowest error)
        [~, winnerLocalIdx] = min(fitnessScores(candidates));
        winnerIdx = candidates(winnerLocalIdx);
        parents(i, :) = population(winnerIdx, :);
    end
end


function offspring = crossover(parents, crossoverRate)
% PLACEHOLDER — full implementation goes in crossover.m
% Single-point crossover on pairs of parents

    [popSize, numGenes] = size(parents);
    offspring = parents;  % start as copy

    for i = 1:2:popSize-1
        if rand() < crossoverRate
            % Pick random crossover point
            point = randi([1, numGenes-1]);
            % Swap genes after crossover point
            offspring(i,   point+1:end) = parents(i+1, point+1:end);
            offspring(i+1, point+1:end) = parents(i,   point+1:end);
        end
    end
end


function offspring = mutate(offspring, mutationRate)
% PLACEHOLDER — full implementation goes in mutate.m
% Gaussian nudge on random genes

    [popSize, numGenes] = size(offspring);

    for i = 1:popSize
        for g = 1:numGenes
            if rand() < mutationRate
                % Small gaussian noise — std dev = 1% of gene range (rough estimate)
                offspring(i, g) = offspring(i, g) + randn() * 0.5;
            end
        end
    end
end