function offspring = crossover(parents, crossoverRate)
% crossover - Applies single-point crossover to produce offspring
%
% INPUT:
%   parents       - matrix [popSize x 70] of selected parent chromosomes
%   crossoverRate - probability that two parents exchange genes (e.g. 0.8 = 80%)
%
% OUTPUT:
%   offspring     - matrix [popSize x 70] of child chromosomes
%
% HOW IT WORKS:
%   Parents are paired up sequentially: (1,2), (3,4), (5,6), ...
%   For each pair, a random crossover point is chosen (e.g. gene 35).
%   Child 1 gets genes 1-35 from Parent 1, genes 36-70 from Parent 2.
%   Child 2 gets genes 1-35 from Parent 2, genes 36-70 from Parent 1.
%   If rand() > crossoverRate, parents are copied unchanged (no crossover).
%
% CONSTRAINT:
%   time_to_peak_min <= time_to_peak_max is enforced after crossover
%   (crossover point could split a min/max pair onto different parents)
%
% Author: Alina
% Project: LIMB - Personalised Motor Rehabilitation

[popSize, numGenes] = size(parents);
offspring = parents;  % start as copy of parents

% -------------------------------------------------------------------------
% CROSSOVER — process pairs (1,2), (3,4), (5,6) ...
% -------------------------------------------------------------------------

for i = 1:2:popSize-1

    if rand() < crossoverRate

        % Pick a random crossover point between gene 1 and gene 69
        % (not at the very ends, so both children get genes from both parents)
        point = randi([1, numGenes-1]);

        % Child 1: parent i up to point, then parent i+1 after point
        offspring(i,   1:point)     = parents(i,   1:point);
        offspring(i,   point+1:end) = parents(i+1, point+1:end);

        % Child 2: parent i+1 up to point, then parent i after point
        offspring(i+1, 1:point)     = parents(i+1, 1:point);
        offspring(i+1, point+1:end) = parents(i,   point+1:end);

    end
    % else: offspring already = parents (no crossover, copied above)

end

% -------------------------------------------------------------------------
% ENFORCE time_to_peak_min <= time_to_peak_max after crossover
% Crossover point could land between gene 4 and 5 (for example),
% giving child 1 a large min from one parent and small max from another
% -------------------------------------------------------------------------

timePeakMinGenes = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69];

for i = 1:length(timePeakMinGenes)
    minIdx = timePeakMinGenes(i);
    maxIdx = minIdx + 1;

    needSwap = offspring(:, minIdx) > offspring(:, maxIdx);
    temp = offspring(needSwap, minIdx);
    offspring(needSwap, minIdx) = offspring(needSwap, maxIdx);
    offspring(needSwap, maxIdx) = temp;
end

end