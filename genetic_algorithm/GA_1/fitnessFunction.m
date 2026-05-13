function fitnessScores = fitnessFunction(population, gaInputPath)
% fitnessFunction - Fitness function for the LIMB Genetic Algorithm
%
% Evaluates how well each chromosome in the population describes the user's
% actual kinematic profile, as observed across their 9 movement recordings.
%
% INPUT:
%   population   - matrix [popSize x 70] of chromosomes to evaluate
%                  each row is one candidate kinematic profile
%   gaInputPath  - path to the patient_X_ga_input.json file produced
%                  by jsonSetup.py (e.g. 'patient_4_ga_input.json')
%
% OUTPUT:
%   fitnessScores - vector [popSize x 1] of fitness values
%                   LOWER = BETTER (fitness = total weighted error + penalties)
%                   0 would mean a perfect match with no constraint violations
%
% HOW FITNESS IS COMPUTED:
%   For each chromosome, for each of the 3 recordings, for each of the 13
%   active joints (wr_pron excluded — unreliable camera signal) — compute
%   the normalised absolute error between the chromosome's gene value and
%   the observed value from the recording. Features are weighted by
%   importance. Then add penalty terms for constraint violations.
%
%   Final score = weighted_error + penalties
%
% NOTE ON wr_pron:
%   Wrist pronation is kept in the chromosome (gene indices 16-20) but
%   excluded from fitness scoring. The palm-normal method is too sensitive
%   to camera viewpoint during arm movement to produce a reliable signal.
%   To be revisited once the vision subsystem provides a better estimate.
%
% GENE ORDER (must match initPopulation.m exactly):
%   Joints 1-4:   sh_flex, sh_abd, el_flex, wr_pron  (5 genes each = 20)
%   Joints 5-14:  thumb through pinky MCP+PIP         (5 genes each = 50)
%   Per joint:    [max_angle, start_angle, smoothness, ttp_min, ttp_max]
%
% Author: Aiden (GA / Fitness Function)
% Project: LIMB - Personalised Motor Rehabilitation

% =========================================================================
% SECTION 1 — CONFIGURATION
% =========================================================================

% --- Feature weights (must sum to 1) ---
% max_angle    : highest weight — defines the user's actual range of motion
% ttp_min/max  : high weight   — captures how fast/slow the user moves
% start_angle  : medium weight — resting position, fairly stable
% smoothness   : lowest weight — noisiest signal, least reliable
W_MAX_ANGLE   = 0.35;
W_TTP_MIN     = 0.20;
W_TTP_MAX     = 0.20;
W_START_ANGLE = 0.15;
W_SMOOTHNESS  = 0.10;
% Total: 1.00

% --- Penalty weights ---
PENALTY_BOUNDS    = 10.0;   % gene value outside its allowed range
PENALTY_TTP_ORDER =  5.0;   % time_to_peak_min > time_to_peak_max
% NOTE: Robot hardware shoulder limit removed — deferred to IK solver stage.

% --- Gene bounds — anatomical limits only (must match initPopulation.m exactly) ---
% Each row: [min, max] for that gene index (1-indexed)
%
% Sources:
%   sh_flex  : -60 (extension) to 180 (full overhead flexion)
%   sh_abd   :   0 (arm at side) to 180 (full abduction overhead)
%   el_flex  :   0 (fully extended) to 160 (full flexion, anatomical limit)
%   wr_pron  :   0 to 180 — EXCLUDED from fitness scoring (unreliable camera signal)
%                gene kept in chromosome for future use; bounds set wide
%   thumb_MCP:   0 to  70 (anatomical MCP flexion limit)
%   thumb_PIP:   0 to  90 (IP joint)
%   index_MCP:   0 to 145 (observed max ~139; anatomical limit ~145)
%   index_PIP:   0 to 120 (anatomical PIP limit)
%   middle_MCP:  0 to 145
%   middle_PIP:  0 to 120
%   ring_MCP:    0 to 145
%   ring_PIP:    0 to 120
%   pinky_MCP:   0 to 145
%   pinky_PIP:   0 to 120
%   start_angle bounds raised to match observed resting positions
%   (finger MCPs rest at 80-120 deg in natural hand posture)
bounds = [
    -60, 180;  % 1:  sh_flex · max_angle
    -60,  60;  % 2:  sh_flex · start_angle
      0,   1;  % 3:  sh_flex · smoothness
    0.3,   7;  % 4:  sh_flex · ttp_min
    0.3,   7;  % 5:  sh_flex · ttp_max
      0, 180;  % 6:  sh_abd  · max_angle
      0,  60;  % 7:  sh_abd  · start_angle
      0,   1;  % 8:  sh_abd  · smoothness
    0.3,   7;  % 9:  sh_abd  · ttp_min
    0.3,   7;  % 10: sh_abd  · ttp_max
      0, 160;  % 11: el_flex · max_angle
      0, 160;  % 12: el_flex · start_angle  (patient rests with arm already bent)
      0,   1;  % 13: el_flex · smoothness
    0.3,   7;  % 14: el_flex · ttp_min
    0.3,   7;  % 15: el_flex · ttp_max
      0, 180;  % 16: wr_pron · max_angle    (EXCLUDED from fitness — unreliable signal)
      0, 180;  % 17: wr_pron · start_angle  (EXCLUDED from fitness — unreliable signal)
      0,   1;  % 18: wr_pron · smoothness
    0.3,   7;  % 19: wr_pron · ttp_min
    0.3,   7;  % 20: wr_pron · ttp_max
      0,  70;  % 21: thumb_MCP · max_angle
      0,  40;  % 22: thumb_MCP · start_angle
      0,   1;  % 23: thumb_MCP · smoothness
    0.2,   7;  % 24: thumb_MCP · ttp_min
    0.2,   7;  % 25: thumb_MCP · ttp_max
      0,  90;  % 26: thumb_PIP · max_angle
      0,  40;  % 27: thumb_PIP · start_angle
      0,   1;  % 28: thumb_PIP · smoothness
    0.2,   7;  % 29: thumb_PIP · ttp_min
    0.2,   7;  % 30: thumb_PIP · ttp_max
      0, 145;  % 31: index_MCP · max_angle
      0, 145;  % 32: index_MCP · start_angle  (rests ~110-120 in natural posture)
      0,   1;  % 33: index_MCP · smoothness
    0.2,   7;  % 34: index_MCP · ttp_min
    0.2,   7;  % 35: index_MCP · ttp_max
      0, 120;  % 36: index_PIP · max_angle
      0,  60;  % 37: index_PIP · start_angle
      0,   1;  % 38: index_PIP · smoothness
    0.2,   7;  % 39: index_PIP · ttp_min
    0.2,   7;  % 40: index_PIP · ttp_max
      0, 145;  % 41: middle_MCP · max_angle
      0, 145;  % 42: middle_MCP · start_angle
      0,   1;  % 43: middle_MCP · smoothness
    0.2,   7;  % 44: middle_MCP · ttp_min
    0.2,   7;  % 45: middle_MCP · ttp_max
      0, 120;  % 46: middle_PIP · max_angle
      0,  60;  % 47: middle_PIP · start_angle
      0,   1;  % 48: middle_PIP · smoothness
    0.2,   7;  % 49: middle_PIP · ttp_min
    0.2,   7;  % 50: middle_PIP · ttp_max
      0, 145;  % 51: ring_MCP · max_angle
      0, 145;  % 52: ring_MCP · start_angle
      0,   1;  % 53: ring_MCP · smoothness
    0.2,   7;  % 54: ring_MCP · ttp_min
    0.2,   7;  % 55: ring_MCP · ttp_max
      0, 120;  % 56: ring_PIP · max_angle
      0,  60;  % 57: ring_PIP · start_angle
      0,   1;  % 58: ring_PIP · smoothness
    0.2,   7;  % 59: ring_PIP · ttp_min
    0.2,   7;  % 60: ring_PIP · ttp_max
      0, 145;  % 61: pinky_MCP · max_angle
      0, 145;  % 62: pinky_MCP · start_angle
      0,   1;  % 63: pinky_MCP · smoothness
    0.2,   7;  % 64: pinky_MCP · ttp_min
    0.2,   7;  % 65: pinky_MCP · ttp_max
      0, 120;  % 66: pinky_PIP · max_angle
      0,  60;  % 67: pinky_PIP · start_angle
      0,   1;  % 68: pinky_PIP · smoothness
    0.2,   7;  % 69: pinky_PIP · ttp_min
    0.2,   7;  % 70: pinky_PIP · ttp_max
];

% Joint names in chromosome order — used to index into the GA input JSON
jointNames = {
    'sh_flex', 'sh_abd', 'el_flex', 'wr_pron', ...
    'thumb_MCP', 'thumb_PIP', ...
    'index_MCP', 'index_PIP', ...
    'middle_MCP', 'middle_PIP', ...
    'ring_MCP', 'ring_PIP', ...
    'pinky_MCP', 'pinky_PIP'
};

numJoints    = 14;
numGenes     = 70;
genesPerJoint = 5;

% Gene offsets within each joint block (1-indexed)
GENE_MAX_ANGLE   = 1;
GENE_START_ANGLE = 2;
GENE_SMOOTHNESS  = 3;
GENE_TTP_MIN     = 4;
GENE_TTP_MAX     = 5;

% =========================================================================
% SECTION 2 — LOAD AND PARSE GA INPUT JSON
% =========================================================================

rawJson   = fileread(gaInputPath);
gaDoc     = jsondecode(rawJson);
% gaDoc has two top-level fields:
%   .biometrics  — upper_arm_length_mm, forearm_length_mm, shoulder_width_mm
%   .recordings  — 9x1 struct array, one entry per recording
% Each recording has: .recording (string) and .joints (struct of 14 joints)
% Each joint has: .max_angle, .start_angle, .smoothness,
%                 .time_to_peak_min, .time_to_peak_max

gaInput       = gaDoc.recordings;
numRecordings = length(gaInput);

% Pre-extract observed values into a clean matrix for speed:
%   observed(r, j, f) = observed value for recording r, joint j, feature f
%   f: 1=max_angle, 2=start_angle, 3=smoothness, 4=ttp_min, 5=ttp_max
% NaN = missing/null in the JSON (camera couldn't observe that joint)

observed = nan(numRecordings, numJoints, genesPerJoint);

for r = 1:numRecordings
    jointData = gaInput(r).joints;
    for j = 1:numJoints
        jName = jointNames{j};
        jData = jointData.(jName);

        observed(r, j, GENE_MAX_ANGLE)   = nullToNaN(jData.max_angle);
        observed(r, j, GENE_START_ANGLE) = nullToNaN(jData.start_angle);
        observed(r, j, GENE_SMOOTHNESS)  = nullToNaN(jData.smoothness);
        observed(r, j, GENE_TTP_MIN)     = nullToNaN(jData.time_to_peak_min);
        observed(r, j, GENE_TTP_MAX)     = nullToNaN(jData.time_to_peak_max);
    end
end

% Feature weights as a vector aligned to gene offset order
featureWeights = [W_MAX_ANGLE, W_START_ANGLE, W_SMOOTHNESS, W_TTP_MIN, W_TTP_MAX];

% Gene ranges for normalisation (so errors across features are comparable)
geneRanges = bounds(:, 2) - bounds(:, 1);  % [70 x 1]

% =========================================================================
% SECTION 3 — EVALUATE FITNESS FOR EACH CHROMOSOME
% =========================================================================

popSize       = size(population, 1);
fitnessScores = zeros(popSize, 1);

for c = 1:popSize

    chromosome = population(c, :);  % 1x70 row vector
    totalError  = 0.0;
    totalPenalty = 0.0;

    % ---------------------------------------------------------------------
    % 3a. BOUNDS PENALTY
    % Any gene outside its allowed range gets penalised proportionally
    % to how far outside it is, normalised by the gene's range.
    % ---------------------------------------------------------------------

    for g = 1:numGenes
        lb = bounds(g, 1);
        ub = bounds(g, 2);
        val = chromosome(g);

        if val < lb
            violation = (lb - val) / geneRanges(g);
            totalPenalty = totalPenalty + PENALTY_BOUNDS * violation;
        elseif val > ub
            violation = (val - ub) / geneRanges(g);
            totalPenalty = totalPenalty + PENALTY_BOUNDS * violation;
        end
    end

    % ---------------------------------------------------------------------
    % 3b. TTP ORDER PENALTY
    % time_to_peak_min must always be <= time_to_peak_max for every joint.
    % Penalise proportionally to how much they are inverted.
    % ---------------------------------------------------------------------

    ttpMinGenes = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69];

    for i = 1:length(ttpMinGenes)
        minIdx = ttpMinGenes(i);
        maxIdx = minIdx + 1;
        ttpMin = chromosome(minIdx);
        ttpMax = chromosome(maxIdx);

        if ttpMin > ttpMax
            % Normalise violation by the ttp_max gene range (0.3 to 5 = 4.7s)
            violation = (ttpMin - ttpMax) / 4.7;
            totalPenalty = totalPenalty + PENALTY_TTP_ORDER * violation;
        end
    end

    % ---------------------------------------------------------------------
    % 3c. SHOULDER HARDWARE PENALTY — REMOVED
    % Deferred to IK solver stage. No robot hardware constraints enforced here.
    % ---------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 3d. FEATURE ERROR — compare chromosome against observed recordings
    %
    % For each recording, for each joint, for each feature:
    %   - Skip if observed value is NaN (camera couldn't see that joint)
    %   - Compute normalised absolute error: |chromosome - observed| / range
    %   - Weight by feature importance
    %   - Average across all valid (recording, joint, feature) combinations
    % ---------------------------------------------------------------------

    % Joint index 4 = wr_pron — excluded from fitness scoring.
    % The palm-normal pronation computation produces an unreliable signal
    % due to camera viewpoint sensitivity. Excluded until a better method
    % is available from the vision subsystem.
    WR_PRON_JOINT_IDX = 4;

    validComparisons = 0;
    weightedErrorSum = 0.0;

    for r = 1:numRecordings
        for j = 1:numJoints

            % Skip wr_pron entirely
            if j == WR_PRON_JOINT_IDX
                continue
            end

            % Base gene index for this joint in the chromosome (1-indexed)
            geneBase = (j - 1) * genesPerJoint;

            for f = 1:genesPerJoint

                obsVal = observed(r, j, f);

                % Skip if the camera didn't capture this joint/feature
                if isnan(obsVal)
                    continue
                end

                % Gene index in chromosome for this joint+feature
                gIdx = geneBase + f;

                chromVal = chromosome(gIdx);
                range    = geneRanges(gIdx);
                weight   = featureWeights(f);

                % Normalised absolute error (0 = perfect, 1 = worst possible)
                normError = abs(chromVal - obsVal) / range;

                weightedErrorSum = weightedErrorSum + weight * normError;
                validComparisons = validComparisons + weight;
            end
        end
    end

    % Average weighted error (guard against all-NaN edge case)
    if validComparisons > 0
        totalError = weightedErrorSum / validComparisons;
    else
        % No valid observations at all — give a high but finite score
        totalError = 1.0;
    end

    % ---------------------------------------------------------------------
    % 3e. COMBINE ERROR AND PENALTIES
    % Fitness = error (0 to ~1) + penalties (0 to large)
    % The GA minimises this value. 0 = perfect chromosome.
    % ---------------------------------------------------------------------

    fitnessScores(c) = totalError + totalPenalty;

end % chromosome loop

end % function


% =========================================================================
% LOCAL HELPER
% =========================================================================

function val = nullToNaN(x)
% Converts empty/null JSON values (which MATLAB jsondecode gives as [])
% into NaN so they can be handled cleanly with isnan() checks.
    if isempty(x)
        val = NaN;
    else
        val = double(x);
    end
end
