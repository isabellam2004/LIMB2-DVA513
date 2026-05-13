function runAllMovements(patientId, patientName, inputFolder)
% runAllMovements - Orchestrator for LIMB per-movement GA profiling
%
% Runs the Genetic Algorithm once for each of the 3 movement types,
% then combines the results into a single patient profile JSON.
%
% INPUT:
%   patientId    - integer patient ID (e.g. 1)
%   patientName  - string patient name (e.g. 'John Doe')
%   inputFolder  - path to folder containing the 3 GA input JSONs
%                  (e.g. '.' for current folder)
%
% USAGE:
%   runAllMovements(1, 'John Doe', '.')
%
% EXPECTS these files in inputFolder:
%   patient_<id>_move1_ga_input.json
%   patient_<id>_move2_ga_input.json
%   patient_<id>_move3_ga_input.json
%
% OUTPUT:
%   Writes: patient_<id>_profile.json to inputFolder
%
% Author: Aiden (GA / Fitness Function)
% Project: LIMB - Personalised Motor Rehabilitation

fprintf('=== LIMB runAllMovements — Patient %d (%s) ===\n', patientId, patientName);

movements = {'move1', 'move2', 'move3'};

results = struct();

for m = 1:3
    movId   = movements{m};
    jsonPath = fullfile(inputFolder, sprintf('patient_%d_%s_ga_input.json', patientId, movId));

    if ~isfile(jsonPath)
        error('[runAllMovements] Input file not found: %s', jsonPath);
    end

    fprintf('\n--- Running GA for %s ---\n', movId);
    [bestChromosome, bestFitness] = runGA(jsonPath, patientId, patientName);

    results.(movId).chromosome = bestChromosome;
    results.(movId).fitness    = bestFitness;
end

% --- Load biometrics from move1 input (same across all movements) ---
move1Path  = fullfile(inputFolder, sprintf('patient_%d_move1_ga_input.json', patientId));
gaDoc      = jsondecode(fileread(move1Path));
if isfield(gaDoc, 'biometrics')
    biometrics = gaDoc.biometrics;
else
    biometrics = struct();
end

% --- Save combined profile in one shot ---
outputPath = fullfile(inputFolder, sprintf('patient_%d_profile.json', patientId));
saveUserProfile(results, bestFitness, patientId, patientName, biometrics, outputPath);

fprintf('\n=== All movements complete. Profile saved. ===\n');

end
